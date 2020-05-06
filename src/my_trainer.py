import itertools
import shutil
from typing import Dict

import torch
from allennlp.training.trainer import *
from allennlp.training import GradientDescentTrainer
from allennlp.nn import util as nn_util
from config import FLAGS
import logging as log

logger = log.getLogger()

class MyTrainer(GradientDescentTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_loss(self, batch, for_training):
        """
        Does a forward pass on the given batches and returns the `loss` value in the result.
        If `for_training` is `True` also applies regularization penalty.
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        if isinstance(batch, torch.Tensor):
            output_dict = self._pytorch_model(batch)
        else:
            output_dict = self._pytorch_model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss

    def train(self) -> Dict[str, Any]: # For the most part a copy of the allennlp method, with support for distinction between inter- and intra-epoch checkpoints
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.
                    if self._distributed:
                        dist.barrier()

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=[self.cuda_device],
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training and removing intra-epoch checkpoints.")
                        self._checkpointer.remove_intra_epoch_checkpoints()
                        break

            if self._master:
                self._tensorboard.log_metrics(
                    train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
                )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._master:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)

            if self._master:
                self._save_checkpoint(epoch)

            # Wait for the master to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

class MyCheckpointer(Checkpointer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._num_temp_serialized_models_to_keep = num_temp_serialized_models_to_keep


    def save_checkpoint(
            self,
            epoch: Union[int, str],
            model_state: Dict[str, Any],
            training_states: Dict[str, Any],
            is_best_so_far: bool,
    ) -> None:
        is_intra_epoch_checkpoint = re.search(".*\..*", str(epoch)) is not None
        if self._serialization_dir is not None:
            model_path = os.path.join(
                self._serialization_dir, "model_state_epoch_{}.th".format(epoch)
            )
            torch.save(model_state, model_path)
            training_path = os.path.join(
                self._serialization_dir, "training_state_epoch_{}.th".format(epoch)
            )
            torch.save({**training_states, "epoch": epoch}, training_path)

            if is_best_so_far and not is_intra_epoch_checkpoint:
                logger.info(
                    f"Best validation performance so far. Copying weights from epoch {str(epoch)} to '{self._serialization_dir}/best.th'."
                )
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

            # Clear out this epoch's intra-epoch-checkpoints
            if not is_intra_epoch_checkpoint:
                self.remove_intra_epoch_checkpoints()

            if (
                    self._num_serialized_models_to_keep is not None
                    and self._num_serialized_models_to_keep >= 0
                    and is_intra_epoch_checkpoint
            ):
                self._serialized_paths.append((time.time(), model_path, training_path))
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = (
                                save_time - self._last_permanent_saved_checkpoint_time
                        )
                        if (
                                time_since_checkpoint_kept
                                > self._keep_serialized_model_every_num_seconds
                        ):
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time
                    if remove_path:
                        for fname in paths_to_remove[1:]:
                            if os.path.isfile(fname):
                                os.remove(fname)

    def remove_intra_epoch_checkpoints(self):
        for i, _ in enumerate(self._serialized_paths):
            paths_to_remove = self._serialized_paths.pop(i)
            for fname in paths_to_remove[1:]:
                if os.path.isfile(fname):
                    os.remove(fname)

