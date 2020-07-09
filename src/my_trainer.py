import itertools
import shutil
from typing import Dict

import torch
from allennlp.training.trainer import *
from allennlp.training import GradientDescentTrainer
from allennlp.nn import util as nn_util
from config import FLAGS
import logging as log

from my_utils.util import TqdmToLogger

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

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = common_util.peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in common_util.gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches = math.ceil(
            len(self.data_loader) / self._num_gradient_accumulation_steps
        )

        # Starting batch (nonzero if restarting mid-way an epoch
        if self.data_loader.dataset.current_permuted_indices is None:
            starting_batch = 0
        else:
            # batches_per_row = len(self.data_loader) / len(self.data_loader.dataset.current_permuted_indices)
            batches_per_row = 1/FLAGS.d_batch
            batches_per_row /= self._num_gradient_accumulation_steps
            starting_batch = math.ceil(batches_per_row * self.data_loader.dataset.row_index)

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            # Nathan added
            tqdm_out = TqdmToLogger(logger, level=logging.INFO)
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches, initial=starting_batch, file=tqdm_out,
                mininterval=FLAGS.log_interval
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        cumulative_batch_group_size = 0
        done_early = False

        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            for batch in batch_group:
                loss = self.batch_loss(batch, for_training=True)
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                loss = loss / len(batch_group)
                if self._opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss += loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch() and self._master:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1))
                    param_norm = torch.norm(param.view(-1)).cpu()
                    self._tensorboard.add_train_scalar(
                        "gradient_update/" + name,
                        update_norm / (param_norm + nn_util.tiny_value_of_dtype(param_norm.dtype)),
                    )
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )

            # Updating tqdm only for the master as the trainers wouldn't have one
            if self._master:
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_postfix_str(description, refresh=False)

            # Log parameter values to Tensorboard (only from the master)
            if self._tensorboard.should_log_this_batch() and self._master:
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch() and self._master:
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                batch_group_size = sum(training_util.get_batch_size(batch) for batch in batch_group)
                cumulative_batch_group_size += batch_group_size
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_group_size / batches_this_epoch
                    logger.info(
                        f"current batch size: {batch_group_size} mean batch size: {average}"
                    )
                    self._tensorboard.add_train_scalar("current_batch_size", batch_group_size)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if (
                    self._model_save_interval is not None
                    and (time.time() - last_save_time > self._model_save_interval)
                    and self._master
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=[self.cuda_device],
        )
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def train(self) -> Dict[
        str, Any]:  # For the most part a copy of the allennlp method, with support for distinction between inter- and intra-epoch checkpoints
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

    # Adapted to also save information about the dataloader, to be able to do a restart mid-epoch
    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        # Parameters

        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
            "data_loader": self.data_loader
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far(),
        )

        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    # Adapted to also save information about the dataloader, to be able to do a restart mid-epoch
    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`

        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        # Returns

        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if (
                self._learning_rate_scheduler is not None
                and "learning_rate_scheduler" in training_state
        ):
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the `training_state` contains a serialized `MetricTracker`.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked `val_metric_per_epoch`.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()
        self.data_loader = training_state['data_loader']
        # To deal with restarts from intra-epoch stops
        if self.data_loader.dataset.chunk_paths or self.data_loader.dataset.current_chunk_path:  # This indicates that we didn't finish with all chunks in the epoch that the training state was in
            epochs_to_add = 0
        else:
            epochs_to_add = 1

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + epochs_to_add
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + epochs_to_add

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return


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
