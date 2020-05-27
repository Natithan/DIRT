import numpy as np
import pickle
import time
import torch
from allennlp.data import Vocabulary
import glob
from pathlib2 import Path
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset

from constants import DECODER_START_TOKEN, READ_ONLY_ROOT
import os
from config import FLAGS, get_my_tokenizer
from tqdm import tqdm
import logging as log
import pandas as pd
import random
corpus_to_data = {
    'wiki': "/cw/working-arwen/damien/datasets/wiki/text/*/*",
    'bookcorpus': "/cw/working-arwen/damien/libs/VL-BERT/data/en_corpus/bc1g.doc",
    'gutenberg': "/cw/working-arwen/nathan/phd/data/pretraining/Gutenberg/*/*.txt"
}


def add_custom_tokens(vocab):
    """
    Add extra tokens needed for the specific encoder-decoder model I am using
    """
    vocab.add_token_to_namespace(
        DECODER_START_TOKEN)  # TODO DECODER: make sure you can directly use the ID for this in the decoder


class SingleDataset():

    def __init__(self, corpus_name, split_name):
        self.id_tensor_path = Path(FLAGS.blob_folder, f'{corpus_name}_{split_name}_ids_tensor').as_posix()
        self.text_path = corpus_to_data[corpus_name]
        self.split_name = split_name
        self.corpus = corpus_name
        self.token_indexer = get_my_tokenizer()

    def get_data(self):
        splitname_no_mini = self.split_name.split("_")[0]

        split_names = [self.split_name.replace(splitname_no_mini,s) for s in ['train','test','val']]
        split_paths = [Path(FLAGS.blob_folder, f'{self.corpus}_{sn}_ids_tensor').as_posix()
                       for sn in split_names]
        assert self.id_tensor_path in split_paths, f"{self.id_tensor_path} is not path of {split_paths}, check spelling"
        if all([os.path.exists(p) for p in split_paths]) and not FLAGS.fresh_data:
            start = time.time()
            print(f'Loading {self.id_tensor_path} ')
            result = torch.load(self.id_tensor_path,map_location='cpu')
            print(f'Loaded {self.id_tensor_path} in {time.time() - start:.2} seconds')
        else:
            result = self._read_data(split_names, split_paths)
        return result

    def _read_data(self,split_names, split_paths):
        log.info(f"Creating and storing splits for {self.corpus}")
        full_tensor = self.get_full_tensor()
        train, val, test = full_tensor[:int(.9 * len(full_tensor))], \
                           full_tensor[int(.9 * len(full_tensor)):int(.95 * len(full_tensor))], \
                           full_tensor[int(.95 * len(full_tensor)):]

        path_to_split = dict(zip(split_paths, [train, test, val]))
        name_to_split = dict(zip(split_names, [train, test, val]))
        for path, split in path_to_split.items():
            torch.save(split, path)
        split_tensor = name_to_split[self.split_name]

        return split_tensor

    def get_full_tensor(self):
        log.info(f"Loading {self.corpus} text data from {self.text_path}.")
        tensor_list = []
        for i, path in tqdm(enumerate(glob.glob(corpus_to_data[self.corpus]))):
            if FLAGS.mini:
                if i > 1:
                    break
            tensor_list += self.text_to_tensor_row(path)
        full_tensor = torch.cat(tensor_list)
        # if self.name == 'bookcorpus':
        #     l = []
        #     for i, line in tqdm(enumerate(open(BOOKCORPUS_DATA_PATH).read().splitlines())):
        #         if FLAGS.mini:
        #             if i > mini_amount:
        #                 break
        #         line = " ".join(line.strip().split())
        #         paragraph += line + " "
        #         if line == "":
        #             l += [{"id": i, "text": paragraph}]
        #             paragraph = ""
        #     full_df = pd.DataFrame(l)
        #     del l
        # elif self.name == 'wiki':
        #     l = []
        #     for i, path in tqdm(enumerate(glob.glob(WIKI_DATA_PATH))):
        #         if FLAGS.mini:
        #             if i > mini_amount:
        #                 break
        #         l += [pd.read_json(path)]
        #     full_df = pd.concat(l)
        # elif self.name == 'gutenberg':
        #     for i, path in tqdm(enumerate(glob.glob(WIKI_DATA_PATH))):
        #         with open(path,'r') as f:
        #             whole_book = f.read()
        #         l += [{"id": i, "text": whole_book}]

        return full_tensor

    def text_to_tensor_row(self, path):
        token_ids_units = []
        log.disable(log.WARNING)
        if self.corpus == 'wiki':
            df = pd.read_json(path, lines=True)
            unflattened_ids = [self.token_indexer.encode(text, add_special_tokens=False) for text in df['text']]
            token_ids_units += unflattened_ids
        elif self.corpus == 'bookcorpus':
            paragraph = ""
            for i, line in tqdm(enumerate(open(path).read().splitlines())):
                if FLAGS.mini:
                    if i > 1000:
                        break
                line = " ".join(line.strip().split())
                paragraph += line + " "
                if line == "":
                    token_ids_units += [self.token_indexer.encode(paragraph, add_special_tokens=False)]
                    paragraph = ""

        elif self.corpus == 'gutenberg':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                whole_book = f.read()
            token_ids_units += [self.token_indexer.encode(whole_book, add_special_tokens=False)]
        log.disable(log.NOTSET)

        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        tensor_list = []
        for token_ids in token_ids_units:
            for i in range(0, len(token_ids), max_raw_seq_length):
                current_sequence = self.token_indexer.prepare_for_model(token_ids[i:i + max_raw_seq_length],
                                                                        truncation_strategy='do_not_truncate',
                                                                        pad_to_max_length=True)['input_ids']
                tensor_list.append(torch.tensor(current_sequence).unsqueeze(
                    0))
        return [torch.cat(tensor_list)]


class ChunkDataSet(IterableDataset):
    def __init__(self,chunk_path):
        self.data = torch.load(chunk_path.as_posix())
        self.data = self.data[torch.randperm(len(self.data))]

    def __iter__(self):
        for row in self.data:
            yield row


class CombinedSplitDataset(IterableDataset):
    def __init__(self, split):
        super().__init__()
        self.token_indexer = get_my_tokenizer()
        self.split_name = split
        self.split_chunks_folder = Path(FLAGS.blob_folder,f'{split}')

    def make_chunks(self):
        total_data_tensor = self.get_data()
        split_into_chunks(self.split_name,total_data_tensor)

    def __iter__(self): #TODO make sure this supports multi-GPU loading with worker_info = torch.utils.data.get_worker_info(): https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        if not self.split_chunks_folder.is_dir():
            self.make_chunks()
        chunk_paths = list(self.split_chunks_folder.glob('*.pt'))
        assert chunk_paths, f"{self.split_chunks_folder} is empty!"
        while chunk_paths:
            chunk_path = chunk_paths.pop(random.randrange(len(chunk_paths)))
            chunk_data = torch.load(chunk_path.as_posix())
            chunk_data = chunk_data[torch.randperm(len(chunk_data))]
            for row in chunk_data:
                yield row

    def __len__(self):
        maybe_mini = '_mini' if FLAGS.mini else ''
        length_blob_path = Path(FLAGS.blob_folder, f'{self.split_name}_tensor_combined{maybe_mini}_length').as_posix()
        if os.path.exists(length_blob_path) and not FLAGS.fresh_data:
            length = torch.load(length_blob_path,map_location='cpu')
        else:
            length = len(self.get_data())
            torch.save(length, length_blob_path)
        return int(length / FLAGS.d_batch)


    def get_data(self):
        blob_path = Path(FLAGS.blob_folder, f'{self.split_name}_tensor_combined').as_posix()
        if os.path.exists(blob_path) and not FLAGS.fresh_data:
            log.info(f'Loading {blob_path} ...')
            start = time.time()
            result = torch.load(blob_path,map_location='cpu')
            log.info(f'Loaded {blob_path} in {time.time() - start:.2} seconds')
        else:
            result = self.combine_data()
            torch.save(result, blob_path)
        return result


    def combine_data(self):
        corpus_names = ['wiki', 'gutenberg', 'bookcorpus']
        return torch.cat(
            [SingleDataset(corpus_name=corpus_name, split_name=self.split_name).get_data()
             for corpus_name in corpus_names])


def get_data_dict():
    '''
    Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    '''
    blob_dir_path = Path(READ_ONLY_ROOT, 'blobs')
    if not os.path.exists(blob_dir_path):
        os.mkdir(blob_dir_path)

    maybe_mini = '_mini' if FLAGS.mini else ''
    train_dataset = CombinedSplitDataset(f'train{maybe_mini}_{FLAGS.max_seq_length}')
    test_dataset = CombinedSplitDataset(f'test{maybe_mini}_{FLAGS.max_seq_length}')
    val_dataset = CombinedSplitDataset(f'val{maybe_mini}_{FLAGS.max_seq_length}')
    return {"train": train_dataset,
            "test": test_dataset,
            "val": val_dataset}


class GutenbergSplitDataset(Dataset):
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
            result = torch.load(self.blob_path,map_location='cpu')
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


def get_data_dict_old():
    '''
    Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    '''
    log.info("Loading old Gutenberg-only data.")
    blob_dir_path = Path(READ_ONLY_ROOT, 'blobs')
    if not os.path.exists(blob_dir_path):
        os.mkdir(blob_dir_path)
    maybe_mini = '_mini' if FLAGS.mini else ''
    train_dataset = GutenbergSplitDataset(Path(FLAGS.pretrain_data_folder, 'train').as_posix(),
                                          Path(blob_dir_path, f'train_tensor{maybe_mini}').as_posix())
    test_dataset = GutenbergSplitDataset(Path(FLAGS.pretrain_data_folder, 'test').as_posix(),
                                         Path(blob_dir_path, f'test_tensor{maybe_mini}').as_posix())
    val_dataset = GutenbergSplitDataset(Path(FLAGS.pretrain_data_folder, 'val').as_posix(),
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
    return {"train": train_dataset,
            "test": test_dataset,
            "val": val_dataset}

def split_into_chunks(split_name,split_tensor):
    nb_chunks = 100
    step = len(split_tensor) // nb_chunks
    split_dir = Path(FLAGS.blob_folder, split_name)
    Path.mkdir(split_dir)
    for i in range(nb_chunks):
        chunk = split_tensor[i * step:min((i + 1) * step, len(split_tensor))].clone()
        path = Path(split_dir, f'{i}.pt').as_posix()
        torch.save(chunk, path)
        del chunk
