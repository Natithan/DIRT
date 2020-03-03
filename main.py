# %% Imports
from __future__ import unicode_literals, print_function
import os

import torch
from torch import nn

from models.wrappers import MLMModelWrapper, MODEL_MAPPING

from pathlib import Path
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
import torch.optim as optim

from absl import app
from config import FLAGS

from text_input_pipeline import GutenbergReader


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
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model],vocab)
    device = torch.device(FLAGS.device_idx if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=10e-6)

    iterator = BasicIterator(batch_size=FLAGS.d_batch)
    iterator.index_with(vocab)
    trainer = Trainer(model=model,  # TODO make sure I can pickup training from interrupted process without errors
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=FLAGS.patience,
                      num_epochs=FLAGS.num_epochs,
                      serialization_dir=run_dir)
    trainer.train()

    model(test_dataset)


if __name__ == '__main__':
    app.run(main)
