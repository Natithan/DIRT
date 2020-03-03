import pickle
from collections import Iterable
import time
import dill as dill
import overrides
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import ArrayField
from pathlib2 import Path

from constants import DECODER_START_TOKEN
import os
from config import FLAGS
import numpy as np

from models.wrappers import TOKENIZER_MAPPING


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO make sure you can directly use the ID for this in the decoder


class GutenbergReader(DatasetReader):

    def __init__(self, token_indexer=None):
        super().__init__(lazy=False)
        self.token_indexer = token_indexer or TOKENIZER_MAPPING[FLAGS.model]


    def text_to_instance(self, token_ids, tags=None):

        fields = {'input_ids': ArrayField(np.array(token_ids),dtype=np.int64 )}

        return Instance(
            fields)


    def _read(self, folder_path):
        total_yields = 0
        max_raw_seq_length = FLAGS.max_seq_length - 2 #Exclusing bos and eos tokens
        for i, file in enumerate(os.scandir(folder_path)):
            if FLAGS.mini:
                if i > 0:
                    break
            with open(file, 'rb') as f:
                running_sequence = []
                nb_sequences = 0

                for j, line in enumerate(f):
                    token_ids = self.token_indexer.encode(line.decode("utf-8", errors='ignore'),add_special_tokens=False,add_prefix_space=True) #False to avoid inserting <s> and </s> tokens around every line, as a sequence is made of multiple lines
                    running_sequence += token_ids
                    if len(running_sequence) >= max_raw_seq_length:
                        current_sequence = running_sequence[:max_raw_seq_length]
                        current_sequence = self.token_indexer.encode(current_sequence,add_special_tokens=True) # Now add start and end tokens
                        running_sequence = running_sequence[max_raw_seq_length:]
                        nb_sequences += 1

                        if FLAGS.mini:
                            if nb_sequences < 2:
                                continue
                            if nb_sequences > 4:
                                break
                        total_yields += 1

                        yield self.text_to_instance(current_sequence)


    def _read_data_folders(self):
        train_dataset = self.read(os.path.join(FLAGS.data_folder, 'train'))
        test_dataset = self.read(os.path.join(FLAGS.data_folder, 'test'))
        val_dataset = self.read(os.path.join(FLAGS.data_folder, 'val'))
        vocab = Vocabulary.from_instances(train_dataset + val_dataset,
                                          max_vocab_size=FLAGS.max_vocab_size)
        # for dataset in (train_dataset, test_dataset, val_dataset):
        #     for idx, instance in enumerate(dataset):
        #         instance.index_fields(vocab)
        #         new_instance = self.objective(instance, self.token_indexer,vocab)
        #         dataset[idx] = new_instance
        add_custom_tokens(vocab)
        return {"train": train_dataset,
                "test": test_dataset,
                "val": val_dataset,
                "vocab": vocab}

    def get_data_dict(self):
        '''
        Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
        '''
        blob_dir_path = Path('blobs')
        if not os.path.exists(blob_dir_path):
            os.mkdir(blob_dir_path)
        maybe_mini = '_mini' if FLAGS.mini else ''
        pickle_names = [name + maybe_mini for name in ('train','test','val','vocab')]
        if all([os.path.exists(Path(blob_dir_path,name)) for name in pickle_names]) and not FLAGS.fresh_data:
            result = {}
            for name in pickle_names:
                with open(Path(blob_dir_path,name), 'rb') as f:
                    start = time.time()
                    result[name.replace(maybe_mini,'')] = pickle.load(f)
                    print(f'Loaded {name} pickle in {time.time() - start:.1f} seconds')
        else:
            result = self._read_data_folders()
            for name in result:
                pickle.dump(result[name],open(Path(blob_dir_path,name + maybe_mini), 'wb'))
        return result
        # return self._read_data_folders()
    # def get_data_dict(self):
    #     '''
    #     Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    #     '''
    #     blob_dir_path = Path('blobs')
    #     if not os.path.exists(blob_dir_path):
    #         os.mkdir(blob_dir_path)
    #     maybe_mini = '_mini' if FLAGS.mini else ''
    #     this_blob_path = Path(blob_dir_path,f'{self.__class__.__name__}_data{maybe_mini}.pkl')
    #
    #     if os.path.exists(this_blob_path):
    #         with open(this_blob_path, 'rb') as f:
    #             return pickle.load(f)
    #     else:
    #         result = self._read_data_folders()
    #         pickle.dump(result,open(this_blob_path, 'wb'))
    #         return result

    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, 'rb') as cache_file:
            instances = dill.load(cache_file)
            for instance in instances:
                yield instance

    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        with open(cache_filename, 'wb') as cache_file:
            dill.dump(instances, cache_file)
