import pickle
from pathlib import Path

from allennlp.data.tokenizers import Token
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from constants import DECODER_START_TOKEN
from objectives import t5_denoise_spans_objective
import os
from config import FLAGS, OBJECTIVE_MAPPING


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(DECODER_START_TOKEN) # TODO make sure you can directly use the ID for this in the decoder


class GutenbergReader(DatasetReader):

    def __init__(self, token_indexers=None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"ids": SingleIdTokenIndexer()}
        self.objective = OBJECTIVE_MAPPING[FLAGS.objective]

    def text_to_instance(self, tokens, tags=None):
        inputs, targets = self.objective(tokens)

        # LabelField: through the AllenNLP iterator, this gets converted to a tensor, rather than a dictionary with a tensor for each of possil=
        input_field = LabelField(inputs, self.token_indexers)
        target_field = LabelField(targets, self.token_indexers)
        fields = {"inputs": input_field,
                  "targets": target_field}

        # if tags:
        #     label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
        #     fields["labels"] = label_field

        return Instance(fields) #TODO figure out what kind of token_indexer I want, and how to pick my vocab_size

    def _read(self, folder_path):
        total_yields = 0
        for i, file in enumerate(os.scandir(folder_path)):
            if FLAGS.mini:
                if i > 0:
                    break
            with open(file,'rb') as f:
                running_sequence = []
                nb_sequences = 0

                for j, line in enumerate(f):
                    words = line.strip().split()
                    running_sequence += words
                    if len(running_sequence) >= FLAGS.max_seq_length:
                        current_sequence = running_sequence[:FLAGS.max_seq_length]
                        running_sequence = running_sequence[FLAGS.max_seq_length:]
                        nb_sequences += 1
                        if FLAGS.mini:
                            if nb_sequences < 2:
                                continue
                            if nb_sequences > 4:
                                break
                        total_yields += 1
                        yield self.text_to_instance([Token(word.decode("utf-8",errors='ignore')) for word  in current_sequence]) #Skipping undecodable characters

    def _read_data_folders(self):
        train_dataset = self.read(os.path.join(FLAGS.data_folder,'train'))
        test_dataset = self.read(os.path.join(FLAGS.data_folder,'test'))
        val_dataset = self.read(os.path.join(FLAGS.data_folder,'val'))
        vocab = Vocabulary.from_instances(train_dataset + val_dataset, max_vocab_size=FLAGS.max_vocab_size)
        add_custom_tokens(vocab)
        return {"train":train_dataset,
                "test":test_dataset,
                "val":val_dataset,
                "vocab":vocab}


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