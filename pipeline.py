# %% Imports
from __future__ import unicode_literals, print_function
import os
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, DistributedSampler

from models.wrappers import MLMModelWrapper, MODEL_MAPPING

from pathlib import Path
import torch.optim as optim

from absl import app
from config import FLAGS

from text_input_pipeline import GutenbergReader
from allennlp.training import GradientDescentTrainer, Checkpointer

from util import cleanup, setup


def get_loader(dataset, distributed):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True) # Shuffle needed for negative sampling
        return DataLoader(dataset,
                            batch_size=FLAGS.d_batch,
                            sampler=sampler)
    else:
        return DataLoader(dataset,
                            batch_size=FLAGS.d_batch,
                            shuffle=True)

def main(_):
    # Create folders and files to store results and configs
    run_dir = Path(FLAGS.model_folder, FLAGS.model, FLAGS.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Store configuration in same folder as logs and model
    flagfile = Path(run_dir, 'flagfile.txt')
    if os.path.exists(flagfile):
        os.remove(flagfile)
    open(flagfile, "x")
    FLAGS.append_flags_into_file(flagfile)

    reader = GutenbergReader()
    data_dict = reader.get_data_dict()
    train_dataset, test_dataset, val_dataset, vocab = (data_dict[key] for key in
                                                       ('train', 'test', 'val', 'vocab'))
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model],
                            vocab)

    distributed_wrapper(train,model, run_dir, train_dataset, val_dataset)

    model(test_dataset)


def train(world_size,rank,model, run_dir, train_dataset, val_dataset):

    # If distributed, this is now in one of the threads. Setup makes sure it is in sync with other threads
    distributed = (world_size > 1)
    print(f'Distributed?: {distributed}')
    if distributed:
        setup(rank, world_size)

    model_device = f'cuda:{FLAGS.device_idxs[0]}' if len(FLAGS.device_idxs) != 0 else 'cpu'
    model.cuda(model_device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    loader = get_loader(train_dataset, distributed)
    val_loader = get_loader(val_dataset, distributed)
    checkpointer = Checkpointer(serialization_dir=run_dir,
                                num_serialized_models_to_keep=FLAGS.num_serialized_models_to_keep)
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=loader,
                                     validation_data_loader=val_loader,
                                     patience=FLAGS.patience,
                                     num_epochs=FLAGS.num_epochs,
                                     serialization_dir=run_dir,
                                     model_save_interval=FLAGS.model_save_interval,
                                     checkpointer=checkpointer,
                                     distributed=distributed,
                                     world_size=FLAGS.max_GPUs,
                                     cuda_device=FLAGS.device_idxs[0])
    trainer.train()

    if distributed:
        cleanup()


def distributed_wrapper(function,*args):
    nb_GPUs = len(FLAGS.device_idxs)
    if nb_GPUs > 1:
        mp.spawn(function,
                 args=(nb_GPUs,) + args,
                 nprocs=nb_GPUs,
                 join=True)
    else:
        function(world_size=0,rank=0)

if __name__ == '__main__':
    app.run(main)
