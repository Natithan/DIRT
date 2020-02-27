import pickle
from pathlib import Path

from allennlp.data.tokenizers import Token
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer, OpenaiTransformerBytePairIndexer

from constants import DECODER_START_TOKEN, MASKING_TOKEN
from objectives import t5_denoise_spans_objective
import os
from config import FLAGS, OBJECTIVE_MAPPING
from allennlp.data.tokenizers.word_splitter import OpenAISplitter
import numpy as np

def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO make sure you can directly use the ID for this in the decoder


class GutenbergReader(DatasetReader):

    def __init__(self, token_indexer=None):
        super().__init__(lazy=False)
        self.splitter = OpenAISplitter()
        self.token_indexer = token_indexer or OpenaiTransformerBytePairIndexer(
            model_path="https://allennlp.s3.amazonaws.com/models/openai-transformer-lm-2018.07.23.tar.gz",
            tokens_to_add=[MASKING_TOKEN])


    def text_to_instance(self, token_ids, tags=None):

        fields = {'input_ids': ArrayField(np.array(token_ids),dtype=np.int32)}

        return Instance(
            fields)


    def _read(self, folder_path): #TODO en route to making tokenizer wrapper: see if needed and then how to deal with bpe adding length to slices
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
                                          max_vocab_size=FLAGS.max_vocab_size)  # TODO fix vocab + openai tokenindexer coop: now vocab size is 2??
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
        # blob_dir_path = Path('blobs')
        # if not os.path.exists(blob_dir_path):
        #     os.mkdir(blob_dir_path)
        # maybe_mini = '_mini' if FLAGS.mini else ''
        # this_blob_path = Path(blob_dir_path,f'{self.__class__.__name__}_data{maybe_mini}.pkl')
        #
        # if os.path.exists(this_blob_path):
        #     with open(this_blob_path, 'rb') as f:
        #         return pickle.load(f)
        # else:
        #     result = self._read_data_folders()
        #     pickle.dump(result,open(this_blob_path, 'wb'))
        #     return result
        return self._read_data_folders()
    # TODO get pickling to work for large data, and add option to not use pickle for small data
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
