# %% Imports
from __future__ import unicode_literals, print_function
import os
from pathlib import Path
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer
import torch.optim as optim
import pickle

from absl import app
from config import FLAGS

from model import FullModel
from text_input_pipeline import GutenbergReader


def main(_):
    run_dir = Path(FLAGS.model_folder, FLAGS.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Store configuration in same folder as logs and model
    flagfile = Path(run_dir, 'flagfile.txt')
    open(flagfile, "x")
    FLAGS.append_flags_into_file(flagfile)
    reader = GutenbergReader() #TODO add COPA task later
    train_dataset, test_dataset, val_dataset, vocab = (reader.get_data_dict()[key] for key in ('train','test','val','vocab'))

    model = FullModel(vocab)
    cuda_device = FLAGS.device_idx
    model = model.cuda(cuda_device)

    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=FLAGS.d_batch, sorting_keys=[("inputs", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model, #TODO make sure I can pickup training from interrupted process without errors
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=FLAGS.patience,
                      num_epochs=FLAGS.num_epochs,
                      serialization_dir=Path(FLAGS.model_folder,FLAGS.run_name),
                      cuda_device=cuda_device)
    trainer.train()


if __name__ == '__main__':
    app.run(main)
