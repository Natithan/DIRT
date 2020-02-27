# %% Imports
from __future__ import unicode_literals, print_function
import os
from wrappers import MLMModelWrapper, MODEL_MAPPING, TOKENIZER_MAPPING

from pathlib import Path
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training import Trainer
import torch.optim as optim
import pickle

from absl import app
from config import FLAGS, CONFIG_MAPPING

from model import FullModel
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
    train_dataset, test_dataset, val_dataset, vocab = (reader.get_data_dict()[key] for key in
                                                       ('train', 'test', 'val', 'vocab'))
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model],vocab) # TODO figure out why unfinetuned pretrained HF roberta works in sandbox, but not here
    cuda_device = FLAGS.device_idx
    model = model.cuda(cuda_device)

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
                      serialization_dir=run_dir,
                      cuda_device=cuda_device)
    trainer.train()

    model(test_dataset)


if __name__ == '__main__':
    app.run(main)
