import pickle
import time
import torch
from allennlp.data import Vocabulary
from pathlib2 import Path
from torch.utils.data import Dataset

from constants import DECODER_START_TOKEN, READ_ONLY_ROOT
import os
from config import FLAGS, get_tokenizer


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO DECODER: make sure you can directly use the ID for this in the decoder


class GutenbergSplitDataset(Dataset):
    def __init__(self, text_data_path, blob_path):
        super().__init__()
        self.token_indexer = get_tokenizer()
        self.text_data_path = text_data_path
        self.blob_path = blob_path
        self.data = self.get_data()

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

    def get_data(self):
        if os.path.exists(self.blob_path) and not FLAGS.fresh_data:
            start = time.time()
            result = torch.load(self.blob_path)
            print(f'Loaded {self.blob_path} in {time.time() - start:.2} seconds')
        else:
            result = self.read_data()
            torch.save(result,self.blob_path)
        return result

    def read_data(self):
        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        number_of_files = len(list(os.scandir(self.text_data_path)))
        tensor_list=[]
        for i, file in enumerate(os.scandir(self.text_data_path)):
            if not FLAGS.mini:
                print(f'Reading file {i} out of {number_of_files} in {self.text_data_path}')
            if FLAGS.mini:
                if i > 0:
                    break
            with open(file, 'rb') as f:
                running_sequence = []
                nb_sequences = 0

                for j, line in enumerate(f):
                    token_ids = self.token_indexer.encode(line.decode("utf-8", errors='ignore'),
                                                          add_special_tokens=False)  # False to avoid inserting <s> and </s> tokens around every line, as a sequence is made of multiple lines
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

                        tensor_list.append(torch.tensor(current_sequence).unsqueeze(0))
        return torch.cat(tensor_list)



def get_data_dict():
    '''
    Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    '''
    blob_dir_path = Path(READ_ONLY_ROOT, 'blobs')
    if not os.path.exists(blob_dir_path):
        os.mkdir(blob_dir_path)
    maybe_mini = '_mini' if FLAGS.mini else ''
    train_dataset = GutenbergSplitDataset(Path(FLAGS.data_folder, 'train').as_posix(),
                                          Path(blob_dir_path,f'train_tensor{maybe_mini}').as_posix())
    test_dataset = GutenbergSplitDataset(Path(FLAGS.data_folder, 'test').as_posix(),
                                          Path(blob_dir_path,f'test_tensor{maybe_mini}').as_posix())
    val_dataset = GutenbergSplitDataset(Path(FLAGS.data_folder, 'test').as_posix(),
                                          Path(blob_dir_path,f'test_tensor{maybe_mini}').as_posix())
    # vocab = Vocabulary.from_instances(train_dataset + val_dataset,
    #                                   max_vocab_size=get_tokenizer().vocab_size)
    # add_custom_tokens(vocab)
    return {"train": train_dataset,
            "test": test_dataset,
            "val": val_dataset}