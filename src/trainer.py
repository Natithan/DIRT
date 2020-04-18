from typing import Dict

import torch
from allennlp.training.trainer import *
from allennlp.training import GradientDescentTrainer
from allennlp.nn import util as nn_util


class MyTrainer(GradientDescentTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)


    def batch_loss(self, batch, for_training):
        """
        Does a forward pass on the given batches and returns the `loss` value in the result.
        If `for_training` is `True` also applies regularization penalty.
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        if isinstance(batch,torch.Tensor):
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

    # Exactly the same as GradientDescentTrainer, minus some info-level logging that I don't want :P
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        # peak_cpu_usage = common_util.peak_memory_mb()
        # logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        # gpu_usage = []
        # for gpu, memory in common_util.gpu_memory_mb().items():
        #     gpu_usage.append((gpu, memory))
            # logger.info(f"GPU {gpu} memory usage MB: {memory}")

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
        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
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
                batch_group_generator_tqdm.set_description(description, refresh=False)

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