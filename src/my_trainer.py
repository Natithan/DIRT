import itertools
import shutil
from typing import Dict

import torch
from allennlp.training.trainer import *
from allennlp.training import GradientDescentTrainer
from allennlp.nn import util as nn_util
from config import FLAGS


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

    # def _train_epoch(self, epoch: int) -> Dict[str, float]:
    #     """
    #     Trains one epoch and returns metrics.
    #     """
    #     logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
    #     peak_cpu_usage = common_util.peak_memory_mb()
    #     logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
    #     gpu_usage = []
    #     for gpu, memory in common_util.gpu_memory_mb().items():
    #         gpu_usage.append((gpu, memory))
    #         logger.info(f"GPU {gpu} memory usage MB: {memory}")
    #
    #     train_loss = 0.0
    #     # Set the model to "train" mode.
    #     self._pytorch_model.train()
    #
    #     # Get tqdm for the training batches
    #     batch_generator = iter(self.data_loader)
    #     batch_group_generator = common_util.lazy_groups_of(
    #         batch_generator, self._num_gradient_accumulation_steps
    #     )
    #
    #     logger.info("Training")
    #
    #     num_training_batches = math.ceil(
    #         len(self.data_loader) / self._num_gradient_accumulation_steps
    #     )
    #     # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
    #     # progress is shown
    #     if self._master:
    #         batch_group_generator_tqdm = Tqdm.tqdm(
    #             batch_group_generator, total=num_training_batches
    #         )
    #     else:
    #         batch_group_generator_tqdm = batch_group_generator
    #
    #     self._last_log = time.time()
    #     last_save_time = time.time()
    #
    #     batches_this_epoch = 0
    #     if self._batch_num_total is None:
    #         self._batch_num_total = 0
    #
    #     histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())
    #
    #     cumulative_batch_group_size = 0
    #     done_early = False
    #     subepoch_size = len(self.data_loader) // FLAGS.nb_subepochs
    #     metrics: Dict[str, Any] = {}
    #
    #     for subepoch in range(FLAGS.nb_subepochs):
    #         subepoch_start = subepoch*subepoch_size
    #         subepoch_end = min(subepoch_start + subepoch_size,len(self.data_loader))
    #         for batch_group in itertools.islice(batch_group_generator_tqdm, subepoch_start, subepoch_end):
    #             if self._distributed:
    #                 # Check whether the other workers have stopped already (due to differing amounts of
    #                 # data in each). If so, we can't proceed because we would hang when we hit the
    #                 # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
    #                 # here because NCCL process groups apparently don't support BoolTensor.
    #                 done = torch.tensor(0, device=self.cuda_device)
    #                 torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
    #                 if done.item() > 0:
    #                     done_early = True
    #                     logger.warning(
    #                         f"Worker {torch.distributed.get_rank()} finishing training early! "
    #                         "This implies that there is an imbalance in your training "
    #                         "data across the workers and that some amount of it will be "
    #                         "ignored. A small amount of this is fine, but a major imbalance "
    #                         "should be avoided. Note: This warning will appear unless your "
    #                         "data is perfectly balanced."
    #                     )
    #                     break
    #
    #             batches_this_epoch += 1
    #             self._batch_num_total += 1
    #             batch_num_total = self._batch_num_total
    #
    #             self.optimizer.zero_grad()
    #
    #             for batch in batch_group:
    #                 loss = self.batch_loss(batch, for_training=True)
    #                 if torch.isnan(loss):
    #                     raise ValueError("nan loss encountered")
    #                 loss = loss / len(batch_group)
    #                 if self._opt_level is not None:
    #                     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                         scaled_loss.backward()
    #                 else:
    #                     loss.backward()
    #                 train_loss += loss.item()
    #
    #             batch_grad_norm = self.rescale_gradients()
    #
    #             # This does nothing if batch_num_total is None or you are using a
    #             # scheduler which doesn't update per batch.
    #             if self._learning_rate_scheduler:
    #                 self._learning_rate_scheduler.step_batch(batch_num_total)
    #             if self._momentum_scheduler:
    #                 self._momentum_scheduler.step_batch(batch_num_total)
    #
    #             if self._tensorboard.should_log_histograms_this_batch() and self._master:
    #                 # get the magnitude of parameter updates for logging
    #                 # We need a copy of current parameters to compute magnitude of updates,
    #                 # and copy them to CPU so large models won't go OOM on the GPU.
    #                 param_updates = {
    #                     name: param.detach().cpu().clone()
    #                     for name, param in self.model.named_parameters()
    #                 }
    #                 self.optimizer.step()
    #                 for name, param in self.model.named_parameters():
    #                     param_updates[name].sub_(param.detach().cpu())
    #                     update_norm = torch.norm(param_updates[name].view(-1))
    #                     param_norm = torch.norm(param.view(-1)).cpu()
    #                     self._tensorboard.add_train_scalar(
    #                         "gradient_update/" + name,
    #                         update_norm / (param_norm + nn_util.tiny_value_of_dtype(param_norm.dtype)),
    #                     )
    #             else:
    #                 self.optimizer.step()
    #
    #             # Update moving averages
    #             if self._moving_average is not None:
    #                 self._moving_average.apply(batch_num_total)
    #
    #             # Update the description with the latest metrics
    #             training_metrics = training_util.get_metrics(
    #                 self.model,
    #                 train_loss,
    #                 batches_this_epoch,
    #                 world_size=self._world_size,
    #                 cuda_device=[self.cuda_device],
    #             )
    #
    #             # Updating tqdm only for the master as the trainers wouldn't have one
    #             if self._master:
    #                 description = training_util.description_from_metrics(training_metrics)
    #                 batch_group_generator_tqdm.set_description(description, refresh=False)
    #
    #             # Log parameter values to Tensorboard (only from the master)
    #             if self._tensorboard.should_log_this_batch() and self._master:
    #                 self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
    #                 self._tensorboard.log_learning_rates(self.model, self.optimizer)
    #
    #                 self._tensorboard.add_train_scalar("loss/loss_train", training_metrics["loss"])
    #                 self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in training_metrics.items()})
    #
    #             if self._tensorboard.should_log_histograms_this_batch() and self._master:
    #                 self._tensorboard.log_histograms(self.model, histogram_parameters)
    #
    #             if self._log_batch_size_period:
    #                 batch_group_size = sum(training_util.get_batch_size(batch) for batch in batch_group)
    #                 cumulative_batch_group_size += batch_group_size
    #                 if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
    #                     average = cumulative_batch_group_size / batches_this_epoch
    #                     logger.info(
    #                         f"current batch size: {batch_group_size} mean batch size: {average}"
    #                     )
    #                     self._tensorboard.add_train_scalar("current_batch_size", batch_group_size)
    #                     self._tensorboard.add_train_scalar("mean_batch_size", average)
    #
    #             # Save model if needed.
    #             if (
    #                     self._model_save_interval is not None
    #                     and (time.time() - last_save_time > self._model_save_interval)
    #                     and self._master
    #             ) or (
    #
    #             ):
    #                 last_save_time = time.time()
    #                 self._save_checkpoint(
    #                     "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
    #                 )
    #         # Stuff that is normally done after each epoch, now after each subepoch
    #
    #         if self._validation_data_loader is not None:
    #             with torch.no_grad():
    #                 # We have a validation set, so compute all the metrics on it.
    #                 val_loss, num_batches = self._validation_loss()
    #
    #                 # It is safe again to wait till the validation is done. This is
    #                 # important to get the metrics right.
    #                 if self._distributed:
    #                     dist.barrier()
    #
    #                 val_metrics = training_util.get_metrics(
    #                     self.model,
    #                     val_loss,
    #                     num_batches,
    #                     reset=True,
    #                     world_size=self._world_size,
    #                     cuda_device=[self.cuda_device],
    #                 )
    #
    #         if self._master:
    #             self._tensorboard.log_metrics(
    #                 training_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
    #             )  # +1 because tensorboard doesn't like 0
    #
    #         for key, value in training_metrics.items():
    #             metrics["training_" + key] = value
    #         for key, value in val_metrics.items():
    #             metrics["validation_" + key] = value
    #         if self._metric_tracker.is_best_so_far():
    #             # Update all the best_ metrics.
    #             # (Otherwise they just stay the same as they were.)
    #             metrics["best_epoch"] = epoch
    #             for key, value in val_metrics.items():
    #                 metrics["best_validation_" + key] = value
    #
    #             self._metric_tracker.best_epoch_metrics = val_metrics
    #
    #         if self._serialization_dir and self._master:
    #             common_util.dump_metrics(
    #                 os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
    #             )
    #
    #     if self._distributed and not done_early:
    #         logger.warning(
    #             f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
    #         )
    #         # Indicate that we're done so that any workers that have remaining data stop the epoch early.
    #         done = torch.tensor(1, device=self.cuda_device)
    #         torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
    #         assert done.item()
    #
    #     # Let all workers finish their epoch before computing
    #     # the final statistics for the epoch.
    #     if self._distributed:
    #         dist.barrier()
    #
    #     training_metrics = training_util.get_metrics(
    #         self.model,
    #         train_loss,
    #         batches_this_epoch,
    #         reset=True,
    #         world_size=self._world_size,
    #         cuda_device=[self.cuda_device],
    #     )
    #     training_metrics["cpu_memory_MB"] = peak_cpu_usage
    #     for (gpu_num, memory) in gpu_usage:
    #         training_metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
    #
    #
    #     return training_metrics


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
        is_intra_epoch_checkpoint = re.search(".*\..*", '.') is not None
        if self._serialization_dir is not None:
            model_path = os.path.join(
                self._serialization_dir, "model_state_epoch_{}.th".format(epoch)
            )
            torch.save(model_state, model_path)
            training_path = os.path.join(
                self._serialization_dir, "training_state_epoch_{}.th".format(epoch)
            )
            torch.save({**training_states, "epoch": epoch}, training_path)

            if is_best_so_far:
                logger.info(
                    "Best validation performance so far. Copying weights to '%s/best.th'.",
                    self._serialization_dir,
                )
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

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

    # def save_temp_checkpoint(
    #     self,
    #     model_state: Dict[str, Any],
    #     training_states: Dict[str, Any],
    #         step
    # ) -> None:
    #     """
    #     Method to be used to save temporary within-epoch checkpoints (to be able to resume interrupted training).
    #     Using a separate method for this purpose allows storing a checkpoint per epoch, not throwing
    #     old epoch checkpoints away, and still being able to resume interrupted training
    #     """
    #     if self._serialization_dir is not None:
    #         model_path = os.path.join(
    #             self._serialization_dir, f"temp_model_state_epoch_{epoch}_step_{step}.th"
    #         )
    #         torch.save(model_state, model_path)
    #         training_path = os.path.join(
    #             self._serialization_dir, f"temp_training_state_epoch_{epoch}_step_{step}.th"
    #         )
    #         torch.save({**training_states}, training_path)
    #
    #         if (
    #             self._num_temp_serialized_models_to_keep is not None
    #             and self._num_temp_serialized_models_to_keep >= 0
    #         ):
    #             self._serialized_paths.append((time.time(), model_path, training_path))
    #             if len(self._serialized_paths) > self._num_temp_serialized_models_to_keep:
    #                 paths_to_remove = self._serialized_paths.pop(0)
    #                 # Check to see if we should keep this checkpoint, if it has been longer
    #                 # then self._keep_serialized_model_every_num_seconds since the last
    #                 # kept checkpoint.
    #                 remove_path = True
    #                 if self._keep_serialized_model_every_num_seconds is not None:
    #                     save_time = paths_to_remove[0]
    #                     time_since_checkpoint_kept = (
    #                         save_time - self._last_permanent_saved_checkpoint_time
    #                     )
    #                     if (
    #                         time_since_checkpoint_kept
    #                         > self._keep_serialized_model_every_num_seconds
    #                     ):
    #                         # We want to keep this checkpoint.
    #                         remove_path = False
    #                         self._last_permanent_saved_checkpoint_time = save_time
    #                 if remove_path:
    #                     for fname in paths_to_remove[1:]:
    #                         if os.path.isfile(fname):
    #                             os.remove(fname)
    #
    # def find_latest_checkpoint(self) -> Tuple[str, str]:
    #     """
    #     Return the location of the latest model and training state files.
    #     If there isn't a valid checkpoint then return None.
    #     """
    #     have_checkpoint = self._serialization_dir is not None and (
    #             any(
    #                 "model_state_epoch_" in x for x in os.listdir(self._serialization_dir)
    #             ) or (
    #         any(
    #             "temp_model_state_epoch_" in x for x in os.listdir(self._serialization_dir))
    #     )
    #     )
    #
    #     if not have_checkpoint:
    #         return None
    #
    #     serialization_files = os.listdir(self._serialization_dir)
    #     model_checkpoints = [x for x in serialization_files if ("model_state_epoch" in x) or ("temp_model_state_epoch_" in x)]
    #     # Get the last checkpoint file.  Epochs are specified as either an
    #     # int (for end of epoch files) or with epoch and timestamp for
    #     # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
    #     found_epochs = [
    #         re.search(r"model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in model_checkpoints
    #     ]
    #     int_epochs: Any = []
    #     for epoch in found_epochs:
    #         pieces = epoch.split(".")
    #         if len(pieces) == 1:
    #             # Just a single epoch without timestamp
    #             int_epochs.append([int(pieces[0]), "0"])
    #         else:
    #             # has a timestamp
    #             int_epochs.append([int(pieces[0]), pieces[1]])
    #     last_epoch = sorted(int_epochs, reverse=True)[0]
    #     if last_epoch[1] == "0":
    #         epoch_to_load = str(last_epoch[0])
    #     else:
    #         epoch_to_load = "{0}.{1}".format(last_epoch[0], last_epoch[1])
    #
    #     model_path = os.path.join(
    #         self._serialization_dir, "model_state_epoch_{}.th".format(epoch_to_load)
    #     )
    #     training_state_path = os.path.join(
    #         self._serialization_dir, "training_state_epoch_{}.th".format(epoch_to_load)
    #     )
    #
    #     return (model_path, training_state_path)
