import numpy as np
import pickle
import time
import torch
from allennlp.data import Vocabulary
import glob
from pathlib2 import Path
from torch.utils.data import Dataset

from constants import DECODER_START_TOKEN, READ_ONLY_ROOT, WIKI_DATA_PATH, DF_PICKLE_PATH
import os
from config import FLAGS, get_my_tokenizer
from tqdm import tqdm
import logging as log
import pandas as pd


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO DECODER: make sure you can directly use the ID for this in the decoder


class WikipediaSplitDataset(Dataset):
    def __init__(self, text_data_path, blob_path):
        super().__init__()
        self.token_indexer = get_my_tokenizer()
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
            torch.save(result, self.blob_path)
        return result

    def read_data(self):
        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        if not os.path.exists(DF_PICKLE_PATH):
            log.info(f"Loading text wiki data from {WIKI_DATA_PATH}")
            l = []
            for i, path in tqdm(enumerate(glob.glob(WIKI_DATA_PATH))):
                if FLAGS.mini:
                    if i > 10:
                        break
                l += [pd.read_json(path)]
            wiki_df = pd.concat(l)
        else:
            if FLAGS.mini:
                df_path = Path(Path(DF_PICKLE_PATH).parent, 'tmp_mini_wiki_df.pkl')
            log.info(f"Loading text wiki data from {df_path}")
            wiki_df = pickle.load(open(df_path, 'rb'))

        tensor_list = []
        for i, article in enumerate(wiki_df['text']):
            token_ids = self.token_indexer.encode(article, add_special_tokens=False)
            running_sequence = []
            running_sequence += token_ids
            for i in range(0, len(token_ids), max_raw_seq_length):
                tensor_list.append(torch.tensor(token_ids[i:i + max_raw_seq_length]).unsqueeze(
                    0))
        return torch.cat(tensor_list)


class GutenbergData(Dataset):
    def __init__(self, text_data_path, blob_path):
        super().__init__()
        self.token_indexer = get_my_tokenizer()
        self.text_data_path = text_data_path
        self.blob_path = blob_path
        self.data = self.get_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def get_split(self,split):
        if os.path.exists(self.blob_path) and not FLAGS.fresh_data:
            start = time.time()
            result = torch.load(self.blob_path)
            print(f'Loaded {self.blob_path} in {time.time() - start:.2} seconds')
        else:
            result = self.read_data()
            torch.save(result, self.blob_path)
        return result

    def read_data(self):
        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        number_of_files = len(list(os.scandir(self.text_data_path)))
        tensor_list = []
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


class CombinedSplitDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.token_indexer = get_my_tokenizer()
        self.split = split
        self.data = self.get_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def get_data(self):
        blob_path = Path(FLAGS.blob_folder,f'{self.split}_tensor_combined')
        if os.path.exists(blob_path) and not FLAGS.fresh_data:
            start = time.time()
            result = torch.load(blob_path)
            print(f'Loaded {blob_path} in {time.time() - start:.2} seconds')
        else:
            result = self.read_data()
            torch.save(result, blob_path)
        return result

    def read_data(self):
        combined_tensor_list = []
        for subfolder in os.listdir(FLAGS.pretrain_data_folder):
            if subfolder == 'wiki':
                full_tensor = self.read_wiki_data()
                split_tensor =
            combined_tensor_list.append(new_tensor)
        return torch.cat(combined_tensor_list)

    def read_wiki_data(self):
        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        split_df = self.get_wiki_split()

        tensor_list = []

        for i, article in enumerate(split_df['text']):
            token_ids = self.token_indexer.encode(article, add_special_tokens=False)
            running_sequence = []
            running_sequence += token_ids
            for i in range(0, len(token_ids), max_raw_seq_length):
                tensor_list.append(torch.tensor(token_ids[i:i + max_raw_seq_length]).unsqueeze(
                    0))
        return torch.cat(tensor_list)

    def get_wiki_split(self):
        split_pickle_path = Path(FLAGS.blob_folder, f'wiki_{self.split}_df.pkl', 'wb')
        if not os.path.exists(split_pickle_path) or FLAGS.fresh_data:
            log.info(f"Creating pandas dataframe splits")
            wiki_df = self.get_wiki_df()

            train, val, test = np.split(wiki_df.sample(frac=1), [int(.9 * len(wiki_df)), int(.95 * len(wiki_df))])
            split_to_name = zip([train, test, val], ['train', 'test', 'val'])
            for split, split_name in split_to_name:
                pickle.dump(split, open(Path(FLAGS.blob_folder, f'wiki{split_name}_df.pkl'), 'wb'))
            split_df = dict(split_to_name)[self.split]

        else:
            split_df = pickle.load(open(split_pickle_path, 'rb'))
        return split_df

    def get_wiki_df(self): #TODO finish this :P
        df_path = DF_PICKLE_PATH
        if FLAGS.mini:
            df_path = Path(Path(DF_PICKLE_PATH).parent, 'tmp_mini_wiki_df.pkl')
        if not os.path.exists(df_path):
            log.info(f"Loading text wiki data from {WIKI_DATA_PATH} to create splits")
            l = []
            for i, path in tqdm(enumerate(glob.glob(WIKI_DATA_PATH))):
                if FLAGS.mini:
                    if i > 10:
                        break
                l += [pd.read_json(path)]
            wiki_df = pd.concat(l)
            pickle.dump(wiki_df,open(df_path, 'wb'))
        else:
            log.info(f"Loading text wiki data from {DF_PICKLE_PATH} to create splits")
            wiki_df = pickle.load(open(df_path, 'rb'))
        return wiki_df


def get_data_dict():
    '''
    Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    '''
    blob_dir_path = Path(READ_ONLY_ROOT, 'blobs')
    if not os.path.exists(blob_dir_path):
        os.mkdir(blob_dir_path)
    maybe_mini = '_mini' if FLAGS.mini else ''

    train_dataset = CombinedSplitDataset('train')
    test_dataset = GutenbergData(Path(FLAGS.data_folder, 'test').as_posix(),
                                 Path(blob_dir_path, f'test_tensor{maybe_mini}').as_posix())
    val_dataset = GutenbergData(Path(FLAGS.data_folder, 'val').as_posix(),
                                Path(blob_dir_path, f'val_tensor{maybe_mini}').as_posix())



    # To reduce validation time
    k = 5000

    perm = torch.randperm(val_dataset.data.size(0))
    idx = perm[:k]
    samples = val_dataset.data[idx]
    val_dataset.data = samples

    # To reduce test time
    perm = torch.randperm(test_dataset.data.size(0))
    idx = perm[:k]
    samples = test_dataset.data[idx]
    test_dataset.data = samples

    sources = [GutenbergData(), WikipediaSplitDataset()]


    train_dataset = CombinedSplitDataset(*(ds.get_split('train') for ds in sources))
    test_dataset = CombinedSplitDataset(*(ds.get_split('test') for ds in sources))
    val_dataset = CombinedSplitDataset(*(ds.get_split('val') for ds in sources))

    return {"train": train_dataset,
            "test": test_dataset,
            "val": val_dataset}
