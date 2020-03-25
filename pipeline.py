# %% Imports
from __future__ import unicode_literals, print_function
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist


from models.wrappers import MLMModelWrapper, MODEL_MAPPING

from pathlib import Path
import torch.optim as optim

from absl import app
from config import FLAGS

from text_input_pipeline import GutenbergReader
from allennlp.training import GradientDescentTrainer, Checkpointer


def main(_):
    setup()
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
    model_device = f'cuda:{FLAGS.device_idxs[0]}' if len(FLAGS.device_idxs) != 0 else 'cpu'
    model.cuda(model_device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    loader = DataLoader(train_dataset,
                        batch_size=FLAGS.d_batch,
                        shuffle=True)  # Shuffle needed for negative sampling
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.d_batch)
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
                                     distributed=True,
                                     world_size=FLAGS.max_GPUs,
                                     cuda_device=FLAGS.device_idxs[0])
    trainer.train()

    model(test_dataset)

def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NUM_NODES'] = FLAGS.world_size
    os.environ['CUDA_DEVICES'] = FLAGS.device_idxs

    # initialize the process group
    dist.init_process_group("gloo", world_size=FLAGS.world_size, rank=FLAGS.rank)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42) #TODO probs add the spawn with nb_processes stuff


if __name__ == '__main__':
    app.run(main)
