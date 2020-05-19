import numpy as np
import pickle
import time
import torch
from allennlp.data import Vocabulary
import glob
from pathlib2 import Path
from torch.utils.data import Dataset

from constants import DECODER_START_TOKEN, READ_ONLY_ROOT
import os
from config import FLAGS, get_my_tokenizer
from tqdm import tqdm
import logging as log
import pandas as pd

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

    def __init__(self, corpus_name, split_name,maybe_mini=''):
        self.maybe_mini = maybe_mini
        self.id_tensor_path = Path(FLAGS.blob_folder,f'{corpus_name}{maybe_mini}_{split_name}_ids_tensor').as_posix()
        self.text_path = corpus_to_data[corpus_name]
        self.split_name = split_name
        self.corpus = corpus_name
        self.token_indexer = get_my_tokenizer()


    def get_data(self):
        split_names = ['train', 'test', 'val']
        split_paths = [Path(FLAGS.blob_folder,f'{self.corpus}{self.maybe_mini}_{sn}_ids_tensor').as_posix()
                       for sn in split_names]
        if all([os.path.exists(p) for p in split_paths]) and not FLAGS.fresh_data:
            start = time.time()
            result = torch.load(self.id_tensor_path)
            print(f'Loaded {self.id_tensor_path} in {time.time() - start:.2} seconds')
        else:
            result = self._read_data()
        return result

    def _read_data(self):
        split_names = ['train', 'test', 'val']
        split_paths = [Path(FLAGS.blob_folder,f'{self.corpus}{self.maybe_mini}_{sn}_ids_tensor').as_posix()
                       for sn in split_names]
        log.info(f"Creating and storing splits for {self.corpus}")
        full_tensor = self.get_full_tensor()
        train, val, test = full_tensor[:int(.9*len(full_tensor))],\
                           full_tensor[int(.9*len(full_tensor)):int(.95*len(full_tensor))],\
                           full_tensor[int(.95*len(full_tensor)):]

        path_to_split = dict(zip(split_paths,[train, test, val]))
        name_to_split = dict(zip(split_names,[train, test, val]))
        for path, split in path_to_split.items():
            torch.save(split, path)
        split_tensor = name_to_split[self.split_name]

        return split_tensor

    def get_full_tensor(self):
        log.info(f"Loading {self.corpus} text data from {self.text_path}.")
        tensor_list = []
        for i,path in tqdm(enumerate(glob.glob(corpus_to_data[self.corpus]))):
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
            df = pd.read_json(path,lines=True)
            unflattened_ids =  [self.token_indexer.encode(text, add_special_tokens=False) for text in df['text']]
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
                with open(path,'r',encoding='utf-8', errors='ignore') as f:
                    whole_book = f.read()
                token_ids_units += [self.token_indexer.encode(whole_book, add_special_tokens=False)]
        log.disable(log.NOTSET)

        max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
        tensor_list = []
        for token_ids in token_ids_units:
            for i in range(0, len(token_ids), max_raw_seq_length):
                current_sequence = self.token_indexer.prepare_for_model(token_ids[i:i + max_raw_seq_length], truncation_strategy='do_not_truncate',
                                                     pad_to_max_length=True)['input_ids']
                tensor_list.append(torch.tensor(current_sequence).unsqueeze(
                    0))
        return [torch.cat(tensor_list)]



# class GutenbergData(Dataset):
#     def __init__(self, text_data_path, blob_path):
#         super().__init__()
#         self.token_indexer = get_my_tokenizer()
#         self.text_data_path = text_data_path
#         self.blob_path = blob_path
#         self.data = self.get_data()
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def get_split(self,split):
#         if os.path.exists(self.blob_path) and not FLAGS.fresh_data:
#             start = time.time()
#             result = torch.load(self.blob_path)
#             print(f'Loaded {self.blob_path} in {time.time() - start:.2} seconds')
#         else:
#             result = self.read_data()
#             torch.save(result, self.blob_path)
#         return result
#
#     def read_data(self):
#         max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
#         number_of_files = len(list(os.scandir(self.text_data_path)))
#         tensor_list = []
#         for i, file in enumerate(os.scandir(self.text_data_path)):
#             if not FLAGS.mini:
#                 print(f'Reading file {i} out of {number_of_files} in {self.text_data_path}')
#             if FLAGS.mini:
#                 if i > 0:
#                     break
#             with open(file, 'rb') as f:
#                 running_sequence = []
#                 nb_sequences = 0
#
#                 for j, line in enumerate(f):
#                     token_ids = self.token_indexer.encode(line.decode("utf-8", errors='ignore'),
#                                                           add_special_tokens=False)  # False to avoid inserting <s> and </s> tokens around every line, as a sequence is made of multiple lines
#                     running_sequence += token_ids
#                     if len(running_sequence) >= max_raw_seq_length:
#                         current_sequence = running_sequence[:max_raw_seq_length]
#                         current_sequence = self.token_indexer.encode(current_sequence,
#                                                                      add_special_tokens=True)  # Now add start and end tokens
#                         running_sequence = running_sequence[max_raw_seq_length:]
#                         nb_sequences += 1
#
#                         if FLAGS.mini:
#                             if nb_sequences < 2:
#                                 continue
#                             if nb_sequences > 4:
#                                 break
#
#                         tensor_list.append(torch.tensor(current_sequence).unsqueeze(0))
#         return torch.cat(tensor_list)


class CombinedSplitDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.token_indexer = get_my_tokenizer()
        self.split_name = split
        self.data = self.get_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def get_data(self):
        maybe_mini = '_mini' if FLAGS.mini else ''
        blob_path = Path(FLAGS.blob_folder,f'{self.split_name}_tensor_combined{maybe_mini}').as_posix()
        if os.path.exists(blob_path) and not FLAGS.fresh_data:
            start = time.time()
            result = torch.load(blob_path)
            print(f'Loaded {blob_path} in {time.time() - start:.2} seconds')
        else:
            result = self.combine_data()
            torch.save(result, blob_path)
        return result

    def combine_data(self):
        maybe_mini = '_mini' if FLAGS.mini else ''
        corpus_names = [f'{c}' for c in ['wiki','gutenberg','bookcorpus']]
        return torch.cat([SingleDataset(corpus_name=corpus_name, split_name=self.split_name,maybe_mini=maybe_mini).get_data()
                          for corpus_name in corpus_names])
    # def _read_data(self):
    #     combined_tensor_list = []
    #     for subfolder in os.listdir(FLAGS.pretrain_data_folder):
    #         if subfolder == 'wiki':
    #             sub_tensor = self.read_wiki_data()
    #             combined_tensor_list.append(sub_tensor)
    #         elif subfolder == 'BookCorpus':
    #             sub_tensor = self.read_bookcorpus_data()
    #             combined_tensor_list.append(sub_tensor)
    #         elif subfolder == 'Gutenberg':
    #             sub_tensor = self.read_gutenberg_data()
    #             combined_tensor_list.append(sub_tensor)
    #     return torch.cat(combined_tensor_list)
    #
    # def read_wiki_data(self):
    #     max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
    #     split_df = self.get_wiki_split()
    #
    #     tensor_list = []
    #
    #     for i, article in enumerate(split_df['text']):
    #         log.disable(log.WARNING) #Supress the warnings about too-long sequences.
    #         token_ids = self.token_indexer.encode(article, add_special_tokens=False)
    #         log.disable(log.NOTSET)
    #         running_sequence = []
    #         running_sequence += token_ids
    #         for i in range(0, len(token_ids), max_raw_seq_length):
    #             current_sequence = self.token_indexer.prepare_for_model(token_ids[i:i + max_raw_seq_length], truncation_strategy='do_not_truncate',
    #                                                  pad_to_max_length=True)['input_ids']
    #             tensor_list.append(torch.tensor(current_sequence).unsqueeze(
    #                 0))
    #     return torch.cat(tensor_list)
    #
    # def get_wiki_split(self):
    #     maybe_mini = '_mini' if FLAGS.mini else ''
    #
    #     split_pickle_path = Path(FLAGS.blob_folder, f'wiki_{self.split}_df_{maybe_mini}.pkl')
    #     if not os.path.exists(split_pickle_path) or FLAGS.fresh_data:
    #         log.info(f"Creating pandas dataframe splits")
    #         wiki_df = self.get_wiki_df()
    #
    #         train, val, test = np.split(wiki_df.sample(frac=1), [int(.9 * len(wiki_df)), int(.95 * len(wiki_df))])
    #         name_to_split = dict(zip(['train', 'test', 'val'],[train, test, val]))
    #         for split_name, split in name_to_split.items():
    #             pickle.dump(split, open(split_pickle_path, 'wb'))
    #         split_df = name_to_split[self.split]
    #     else:
    #         split_df = pickle.load(open(split_pickle_path, 'rb'))
    #     return split_df
    #
    # def get_wiki_df(self):
    #     df_path = WIKI_DF_PICKLE_PATH
    #     if FLAGS.mini:
    #         df_path = Path(Path(WIKI_DF_PICKLE_PATH).parent, 'text_mini.pkl')
    #     if not os.path.exists(df_path):
    #         log.info(f"Loading text wiki data from {WIKI_DATA_PATH} to create splits")
    #         l = []
    #         for i, path in tqdm(enumerate(glob.glob(WIKI_DATA_PATH))):
    #             if FLAGS.mini:
    #                 if i > 10:
    #                     break
    #             l += [pd.read_json(path)]
    #         wiki_df = pd.concat(l)
    #         pickle.dump(wiki_df,open(df_path, 'wb'))
    #     else:
    #         log.info(f"Loading text wiki data from {WIKI_DF_PICKLE_PATH} to create splits")
    #         wiki_df = pickle.load(open(df_path, 'rb'))
    #     return wiki_df
    #
    # def read_gutenberg_data(self):
    #     maybe_mini = '_mini' if FLAGS.mini else ''
    #     blob_path = Path(FLAGS.blob_folder, f'Gutenberg_{self.split}_tensor{maybe_mini}').as_posix()
    #     if os.path.exists(blob_path) and not FLAGS.fresh_data:
    #         start = time.time()
    #         result = torch.load(blob_path)
    #         print(f'Loaded {blob_path} in {time.time() - start:.2} seconds')
    #     else:
    #         result = self.read_gutenberg_text()
    #         torch.save(result, blob_path)
    #     return result
    #
    # def read_gutenberg_text(self):
    #     max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
    #     text_data_path = Path(FLAGS.pretrain_data_folder, 'Gutenberg',self.split).as_posix()
    #     number_of_files = len(list(os.scandir(text_data_path)))
    #     tensor_list = []
    #     for i, file in enumerate(os.scandir(text_data_path)):
    #         if not FLAGS.mini:
    #             print(f'Reading file {i} out of {number_of_files} in {text_data_path}')
    #         if FLAGS.mini:
    #             if i > 0:
    #                 break
    #         with open(file, 'rb') as f:
    #             running_sequence = []
    #             nb_sequences = 0
    #
    #             for j, line in enumerate(f):
    #                 token_ids = self.token_indexer.encode(line.decode("utf-8", errors='ignore'),
    #                                                       add_special_tokens=False)  # False to avoid inserting <s> and </s> tokens around every line, as a sequence is made of multiple lines
    #                 running_sequence += token_ids
    #                 if len(running_sequence) >= max_raw_seq_length:
    #
    #                     current_sequence = self.token_indexer.prepare_for_model(running_sequence[:max_raw_seq_length],
    #                                                                             truncation_strategy='do_not_truncate',
    #                                                                             pad_to_max_length=True)['input_ids']
    #                     running_sequence = running_sequence[max_raw_seq_length:]
    #                     nb_sequences += 1
    #
    #                     if FLAGS.mini:
    #                         if nb_sequences < 2:
    #                             continue
    #                         if nb_sequences > 4:
    #                             break
    #
    #                     tensor_list.append(torch.tensor(current_sequence).unsqueeze(0))
    #     return torch.cat(tensor_list)
    #
    #
    # def read_bookcorpus_data(self):
    #     max_raw_seq_length = FLAGS.max_seq_length - 2  # Exclusing bos and eos tokens
    #     split_df = self.get_bookcorpus_split()
    #
    #     tensor_list = []
    #
    #     for i, article in enumerate(split_df['text']):
    #         log.disable(log.WARNING) #Supress the warnings about too-long sequences.
    #         token_ids = self.token_indexer.encode(article, add_special_tokens=False)
    #         log.disable(log.NOTSET)
    #         running_sequence = []
    #         running_sequence += token_ids
    #         for i in range(0, len(token_ids), max_raw_seq_length):
    #             current_sequence = self.token_indexer.prepare_for_model(token_ids[i:i + max_raw_seq_length], truncation_strategy='do_not_truncate',
    #                                                  pad_to_max_length=True)['input_ids']
    #             tensor_list.append(torch.tensor(current_sequence).unsqueeze(
    #                 0))
    #     return torch.cat(tensor_list)
    #
    # def get_bookcorpus_split(self):
    #     maybe_mini = '_mini' if FLAGS.mini else ''
    #
    #     split_pickle_path = Path(FLAGS.blob_folder, f'bookcorpus_{self.split}_df_{maybe_mini}.pkl')
    #     if not os.path.exists(split_pickle_path) or FLAGS.fresh_data:
    #         log.info(f"Creating pandas dataframe splits for BookCorpus")
    #         bc_df = self.get_bookcorpus_df()
    #
    #         train, val, test = np.split(bc_df.sample(frac=1), [int(.9 * len(bc_df)), int(.95 * len(bc_df))])
    #         name_to_split = dict(zip(['train', 'test', 'val'],[train, test, val]))
    #         for split_name, split in name_to_split.items():
    #             pickle.dump(split, open(split_pickle_path, 'wb'))
    #         split_df = name_to_split[self.split]
    #
    #     else:
    #         split_df = pickle.load(open(split_pickle_path, 'rb'))
    #     return split_df
    #
    # def get_bookcorpus_df(self):
    #     if FLAGS.mini:
    #         df_path = Path(Path(BOOKCORPUS_DF_PICKLE_PATH).parent, 'bookcorpus_mini.pkl')
    #     else:
    #         df_path = BOOKCORPUS_DF_PICKLE_PATH
    #     if not os.path.exists(df_path):
    #         log.info(f"Loading text bookcorpus data from {BOOKCORPUS_DATA_PATH} to create splits")
    #         l = []
    #         for i, line in tqdm(enumerate(open(BOOKCORPUS_DATA_PATH).read().splitlines())):
    #             if FLAGS.mini:
    #                 if i > 100:
    #                     break
    #             line = " ".join(line.strip().split())
    #             paragraph += line + " "
    #             if line == "":
    #                 l += [{"id": i, "text": paragraph}]
    #                 paragraph = ""
    #         bc_df = pd.DataFrame(l)
    #         del l
    #         pickle.dump(bc_df,open(df_path, 'wb'))
    #     else:
    #         log.info(f"Loading text bookcorpus data from {BOOKCORPUS_DF_PICKLE_PATH} to create splits")
    #         bc_df = pickle.load(open(df_path, 'rb'))
    #     return bc_df

def get_data_dict():
    '''
    Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    '''
    blob_dir_path = Path(READ_ONLY_ROOT, 'blobs')
    if not os.path.exists(blob_dir_path):
        os.mkdir(blob_dir_path)

    train_dataset = CombinedSplitDataset('train')
    test_dataset = CombinedSplitDataset('test')
    val_dataset = CombinedSplitDataset('val')
    return {"train": train_dataset,
            "test": test_dataset,
            "val": val_dataset}
    # test_dataset = GutenbergData(Path(FLAGS.pretrain_data_folder, 'test').as_posix(),
    #                              Path(blob_dir_path, f'test_tensor{maybe_mini}').as_posix())
    # val_dataset = GutenbergData(Path(FLAGS.pretrain_data_folder, 'val').as_posix(),
    #                             Path(blob_dir_path, f'val_tensor{maybe_mini}').as_posix())



    # # To reduce validation time
    # k = 5000
    #
    # perm = torch.randperm(val_dataset.data.size(0))
    # idx = perm[:k]
    # samples = val_dataset.data[idx]
    # val_dataset.data = samples
    #
    # # To reduce test time
    # perm = torch.randperm(test_dataset.data.size(0))
    # idx = perm[:k]
    # samples = test_dataset.data[idx]
    # test_dataset.data = samples
    #
    # sources = [GutenbergData(), WikipediaSplitDataset()]
    #
    #
    # train_dataset = CombinedSplitDataset(*(ds.get_split('train') for ds in sources))
    # test_dataset = CombinedSplitDataset(*(ds.get_split('test') for ds in sources))
    # val_dataset = CombinedSplitDataset(*(ds.get_split('val') for ds in sources))
    #
    # return {"train": train_dataset,
    #         "test": test_dataset,
    #         "val": val_dataset}

