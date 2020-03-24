import pickle
from collections import Iterable
import time
import dill as dill
import overrides
import torch
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import ArrayField
from pathlib2 import Path
from torch.utils.data import Dataset

from constants import DECODER_START_TOKEN
import os
from config import FLAGS, TOKENIZER_MAPPING
import numpy as np


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO DECODER: make sure you can directly use the ID for this in the decoder


class GutenbergSplitDataset(Dataset):
    def __init__(self, folder_path, token_indexer=None):
        super().__init__()
        self.token_indexer = token_indexer or TOKENIZER_MAPPING[FLAGS.model]
        self.folder_path = folder_path

    def __iter__(self):
        total_yields = 0
        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        for i, file in enumerate(os.scandir(self.folder_path)):
            if FLAGS.mini:
                if i > 0:
                    break
            with open(file, 'rb') as f:
                running_sequence = []
                nb_sequences = 0

                for j, line in enumerate(f):
                    token_ids = self.token_indexer.encode(line.decode("utf-8", errors='ignore'),
                                                          add_special_tokens=False,
                                                          add_prefix_space=True)  # False to avoid inserting <s> and </s> tokens around every line, as a sequence is made of multiple lines
                    running_sequence += token_ids
                    if len(running_sequence) >= max_raw_seq_length:
                        current_sequence = running_sequence[:max_raw_seq_length]
                        current_sequence = self.token_indexer.encode(current_sequence,
                                                                     add_special_tokens=True)  # Now add start and end tokens
                        running_sequence = running_sequence[max_raw_seq_length:]
                        nb_sequences += 1

                        if FLAGS.mini:
                            if nb_sequences < 2:
                                continue
                            if nb_sequences > 4:
                                break
                        total_yields += 1

                        yield {'input_ids':torch.tensor(current_sequence)}


class GutenbergReader:

    def __init__(self, token_indexer=None):
        self.token_indexer = token_indexer or TOKENIZER_MAPPING[FLAGS.model]

    def _read_data_folders(self):
        train_dataset = GutenbergSplitDataset(os.path.join(FLAGS.data_folder, 'train'), self.token_indexer)
        test_dataset = GutenbergSplitDataset(os.path.join(FLAGS.data_folder, 'test'), self.token_indexer)
        val_dataset = GutenbergSplitDataset(os.path.join(FLAGS.data_folder, 'val'), self.token_indexer)
        dummy_vocab = Vocabulary()
        # vocab = Vocabulary.from_instances(train_dataset + val_dataset,
        #                                   max_vocab_size=TOKENIZER_MAPPING[FLAGS.tokenizer].vocab_size)
        # add_custom_tokens(vocab)
        return {"train": list(train_dataset),
                "test": list(test_dataset),
                "val": list(val_dataset),
                "vocab": dummy_vocab}

    def get_data_dict(self):
        '''
        Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
        '''
        blob_dir_path = Path('blobs')
        if not os.path.exists(blob_dir_path):
            os.mkdir(blob_dir_path)
        maybe_mini = '_mini' if FLAGS.mini else ''
        pickle_names = [name + maybe_mini for name in ('train', 'test', 'val', 'vocab')]
        if all([os.path.exists(Path(blob_dir_path, name)) for name in pickle_names]) and not FLAGS.fresh_data:
            result = {}
            for name in pickle_names:
                with open(Path(blob_dir_path, name), 'rb') as f:
                    start = time.time()
                    result[name.replace(maybe_mini, '')] = pickle.load(f)
                    print(f'Loaded {name} pickle in {time.time() - start:.1f} seconds')
        else:
            result = self._read_data_folders()
            for name in result:
                pickle.dump(result[name], open(Path(blob_dir_path, name + maybe_mini), 'wb'))
        return result