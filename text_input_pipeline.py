import pickle
from pathlib import Path

from allennlp.data.tokenizers import Token
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, OpenaiTransformerBytePairIndexer

from constants import DECODER_START_TOKEN, MASKING_TOKEN
from objectives import t5_denoise_spans_objective
import os
from config import FLAGS, OBJECTIVE_MAPPING
from allennlp.data.tokenizers.word_splitter import OpenAISplitter


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO make sure you can directly use the ID for this in the decoder


class GutenbergReader(DatasetReader):

    def __init__(self, token_indexers=None):
        super().__init__(lazy=False)
        self.splitter = OpenAISplitter()
        self.token_indexers = token_indexers or {"ids": OpenaiTransformerBytePairIndexer(
            model_path="https://allennlp.s3.amazonaws.com/models/openai-transformer-lm-2018.07.23.tar.gz",
            tokens_to_add=[MASKING_TOKEN])}
        self.objective = OBJECTIVE_MAPPING[FLAGS.objective]

    def bp_len(self, token_list):
        return len(
            [bp_token for token in token_list for bp_token in self.token_indexers['ids'].byte_pair_encode(token)])

    def text_to_instance(self, tokens, tags=None):
        inputs, targets = self.objective(tokens)

        input_field = TextField(inputs, self.token_indexers)
        target_field = TextField(targets, self.token_indexers)
        fields = {"inputs": input_field,
                  "targets": target_field}

        # if tags:
        #     label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
        #     fields["labels"] = label_field

        return Instance(
            fields)  # TODO make sure no mismatch between masks that get converted to single BPE token, and the words they cover in targets


    def _read(self, folder_path):
        total_yields = 0
        for i, file in enumerate(os.scandir(folder_path)):
            if FLAGS.mini:
                if i > 0:
                    break
            with open(file, 'rb') as f:
                running_sequence = []
                nb_sequences = 0

                for j, line in enumerate(f):
                    tokens = self.splitter.split_words(
                        line.decode("utf-8", errors='ignore'))  # Skipping undecodable characters
                    running_sequence += tokens
                    bp_overflow_amount = self.bp_len(
                        running_sequence) - FLAGS.max_seq_length  # TODO this might be too slow to be workable for large data

                    if bp_overflow_amount >= 0:
                        for t_idx in range(len(tokens)):
                            if self.bp_len(tokens[t_idx:]) < bp_overflow_amount:
                                cutoff_idx = len(running_sequence) - len(tokens) + t_idx - 1
                                break
                        current_sequence = running_sequence[:cutoff_idx]
                        running_sequence = running_sequence[cutoff_idx:]
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
        for dataset in (train_dataset, test_dataset, val_dataset):
            for instance in dataset:
                instance.index_fields(vocab)
        # TODO add masking at this point
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
