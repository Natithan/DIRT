from collections import OrderedDict

import numpy as np
import pickle
import time
import torch
from allennlp.data import Vocabulary
import glob
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset

from constants import DECODER_START_TOKEN, STORAGE_ROOT
import os
from config import FLAGS, get_my_tokenizer
from tqdm import tqdm
import logging as log
import pandas as pd
import random
import nltk
OBJECTIVE_TO_DATA_FORMAT = OrderedDict(
    [
        ("t5_mlm", "single_segment_sequences",),
        ("simple_mlm", "single_segment_sequences",),
        ("albert_mlm_sop", "pairwise_segment_sequences",),
    ]
)
BLOB_SUBFOLDER= Path(FLAGS.blob_folder, OBJECTIVE_TO_DATA_FORMAT[FLAGS.objective])

nltk.download('punkt', download_dir=STORAGE_ROOT)
nltk.data.path.append(STORAGE_ROOT)
SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

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
        self.id_tensor_path = Path(BLOB_SUBFOLDER, f'{corpus_name}_{split_name}_ids_tensor').as_posix()
        self.text_path = corpus_to_data[corpus_name]
        self.split_name = split_name
        self.corpus = corpus_name
        self.token_indexer = get_my_tokenizer()

    def get_data(self):
        actual_split = self.split_name.split("_")[1]
        assert actual_split in ['train', 'test', 'val']

        split_names = [self.split_name.replace(actual_split, s) for s in ['train', 'test', 'val']]
        split_paths = [Path(BLOB_SUBFOLDER, f'{self.corpus}_{sn}_ids_tensor').as_posix()
                       for sn in split_names]
        assert self.id_tensor_path in split_paths, f"{self.id_tensor_path} is not path of {split_paths}, check spelling"
        if all([os.path.exists(p) for p in split_paths]) and not FLAGS.fresh_data:
            start = time.time()
            print(f'Loading {self.id_tensor_path} ')
            result = torch.load(self.id_tensor_path, map_location='cpu')
            print(f'Loaded {self.id_tensor_path} in {time.time() - start:.2} seconds')
        else:
            result = self._read_data(split_names, split_paths)
        return result

    def _read_data(self, split_names, split_paths):
        log.info(f"Creating and storing splits for {self.corpus}")
        full_tensor, sentence_order_list = self.get_full_tensor()
        train, val, test = full_tensor[:int(.9 * len(full_tensor))], \
                           full_tensor[int(.9 * len(full_tensor)):int(.95 * len(full_tensor))], \
                           full_tensor[int(.95 * len(full_tensor)):]
        train_order_labels, val_order_labels, test_order_labels = sentence_order_list[:int(.9 * len(full_tensor))], \
                                                                  sentence_order_list[int(.9 * len(full_tensor)):int(
                                                                      .95 * len(full_tensor))], \
                                                                  sentence_order_list[int(.95 * len(full_tensor)):]

        path_to_split = dict(zip(split_paths, [{'ids': train, 'order_label': train_order_labels},
                                               {'ids': test, 'order_label': test_order_labels},
                                               {'ids': val, 'order_label': val_order_labels}]))
        name_to_split = dict(zip(split_names, [{'ids': train, 'order_label': train_order_labels},
                                               {'ids': test, 'order_label': test_order_labels},
                                               {'ids': val, 'order_label': val_order_labels}]))
        for path, split in path_to_split.items():
            torch.save(split, path)
        split_tensor = name_to_split[self.split_name]

        return split_tensor

    def get_full_tensor(self):
        log.info(f"Loading a fraction {FLAGS.pretrain_data_fraction} of {self.corpus} text data from {self.text_path}.")
        tensor_list = []
        sentence_order_list = []
        all_text_files = glob.glob(corpus_to_data[self.corpus])
        for i, path in tqdm(enumerate(all_text_files)):
            if self.corpus in ['wiki', 'gutenberg']:
                if i > len(all_text_files) * FLAGS.pretrain_data_fraction:
                    break
            ids_and_order_labels = self.text_to_tensor_rows(path)
            tensor_list += ids_and_order_labels['ids']
            sentence_order_list += ids_and_order_labels['order_label']
        full_tensor = torch.cat(tensor_list)

        return full_tensor, sentence_order_list

    # def text_to_tensor_rows(self, path):
    #     # A 'document' is a chunk of text that belongs to each other (e.g. one wikipedia page, one bookcorpus paragraph
    #     # (as bookcorpus is a bunch of mixed paragraphs), or one whole gutenberg book
    #     documents = []
    #     log.disable(log.WARNING)
    #     if self.corpus == 'wiki':
    #         df = pd.read_json(path, lines=True)
    #         unflattened_ids = [self.token_indexer.encode(text, add_special_tokens=False) for text in df['text']]
    #         documents += unflattened_ids
    #     elif self.corpus == 'bookcorpus':
    #         paragraph = ""
    #         full_file = open(path).read().splitlines()
    #         full_length = len(full_file)
    #         for i, line in tqdm(enumerate(full_file)):
    #             if i > full_length * FLAGS.pretrain_data_fraction:
    #                 break
    #             line = " ".join(line.strip().split())
    #             paragraph += line + " "
    #             if line == "":
    #                 documents += [self.token_indexer.encode(paragraph, add_special_tokens=False)]
    #                 paragraph = ""
    #
    #     elif self.corpus == 'gutenberg':
    #         with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    #             whole_book = f.read()
    #         documents += [self.token_indexer.encode(whole_book, add_special_tokens=False)]
    #     log.disable(log.NOTSET)
    #
    #     tensor_list = []
    #     sentence_order_list = []
    #     # We split each document into sentences when doing Sentence Order Prediction, to make the task not too easy
    #     if FLAGS.objective == 'albert_mlm_sop':
    #         documents = self.split_into_sentences(documents)
    #         max_pairwise_seq_length = FLAGS.max_seq_length - 3  # Excluding [CLS] at start, [SEP] in between and [SEP] again at end
    #
    #         # 10% of the time: have shorter sequences
    #         max_pairwise_seq_length = random.randint(2,
    #                               max_pairwise_seq_length) if (random.random() < 0.1) else max_pairwise_seq_length
    #         current_chunk = []
    #         current_length = 0
    #         for document in documents:
    #             j = 0
    #             while j < len(document):
    #                 sentence = document[j]
    #                 # # Last sentence: just attach to current chunk if it fits the length
    #                 # if (j == len(document) - 1) and (current_length + len(sentence) <= max_pairwise_seq_length):
    #                 #     current_chunk.append(sentence)
    #                 #     current_length += len(sentence)
    #                 if current_length + len(sentence) > max_pairwise_seq_length:
    #                     if not current_chunk:
    #                         # This means that the sentence length itself is bigger than max_pairwise_seq_length
    #                         # In this case
    #                     end_a = random.randint(1, max(1, len(current_chunk) - 1))
    #                     tokens_a = [token_id for s in current_chunk[:end_a] for token_id in s]
    #                     tokens_b = [token_id for s in current_chunk[end_a:] for token_id in s]
    #                     assert len(tokens_a) > 0
    #                     assert len(tokens_b) > 0
    #                     in_order = random.random() > 0.5
    #                     if not in_order:
    #                         tokens_a, tokens_b = tokens_b, tokens_a
    #                     current_sequence = self.token_indexer.prepare_for_model(tokens_a, tokens_b,
    #                                                                             truncation_strategy='do_not_truncate',
    #                                                                             pad_to_max_length=True)['input_ids']
    #                     tensor_list.append(torch.tensor(current_sequence).unsqueeze(
    #                         0))
    #                     sentence_order_list.append(int(in_order))
    #                     current_chunk = []
    #                     current_length = 0
    #                     max_pairwise_seq_length = random.randint(1,
    #                               max_pairwise_seq_length) if (random.random() < 0.1) else max_pairwise_seq_length
    #
    #                 else:
    #                     current_chunk.append(sentence)
    #                     current_length += len(sentence)
    #     else:
    #         max_raw_seq_length = FLAGS.max_seq_length - 2  # Excluding [CLS] at start and [SEP] at end
    #         for token_ids in documents:
    #             for i in range(0, len(token_ids), max_raw_seq_length):
    #                 current_sequence = self.token_indexer.prepare_for_model(token_ids[i:i + max_raw_seq_length],
    #                                                                         truncation_strategy='do_not_truncate',
    #                                                                         pad_to_max_length=True)['input_ids']
    #                 tensor_list.append(torch.tensor(current_sequence).unsqueeze(
    #                     0))
    #
    #     return [torch.cat(tensor_list).to(torch.int32)], sentence_order_list

    # Adapted from https://github.com/google-research/albert/blob/7d8e66b191838ed8f4caf414e92a25e7f363d460/create_pretraining_data.py#L602
    def text_to_tensor_rows(self, path):
        """Creates tensor rows and order labels for the text in the given path."""
        do_SOP = (FLAGS.objective == 'albert_mlm_sop')
        all_documents = []
        log.disable(log.WARNING)
        if self.corpus == 'wiki':
            df = pd.read_json(path, lines=True)
            # sentences = self.split_into_sentences(text)
            unflattened_ids = [[self.token_indexer.encode(sentence, add_special_tokens=False)
                                for sentence in self.split_into_sentences(text)]
                               for text in df['text']
                               ] if do_SOP else [self.token_indexer.encode(text, add_special_tokens=False) for text in
                                                 df['text']]
            all_documents += unflattened_ids
        elif self.corpus == 'bookcorpus':
            paragraph = ""
            full_file = open(path).read().splitlines()
            full_length = len(full_file)
            for i, line in tqdm(enumerate(full_file)):
                if i > full_length * FLAGS.pretrain_data_fraction:
                    break
                line = " ".join(line.strip().split())
                paragraph += line + " "
                if line == "":
                    all_documents += [[self.token_indexer.encode(sentence, add_special_tokens=False)
                                       for sentence in self.split_into_sentences(paragraph)]] if do_SOP else \
                        [self.token_indexer.encode(paragraph, add_special_tokens=False)]
                    paragraph = ""

        elif self.corpus == 'gutenberg':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                whole_book = f.read()
            all_documents += [[self.token_indexer.encode(sentence, add_special_tokens=False)
                               for sentence in self.split_into_sentences(whole_book)]] if do_SOP else \
                [self.token_indexer.encode(whole_book, add_special_tokens=False)]
        log.disable(log.NOTSET)

        tensor_list = []
        sentence_order_list = []
        if FLAGS.objective == 'albert_mlm_sop':
            # To use as not-in-order segment for one-segment documents
            global previous_segment

            # Account for [CLS], [SEP], [SEP]
            max_num_tokens = FLAGS.max_seq_length - 3

            # We *usually* want to fill up the entire sequence since we are padding
            # to `max_seq_length` anyways, so short sequences are generally wasted
            # computation. However, we *sometimes*
            # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
            # sequences to minimize the mismatch between pre-training and fine-tuning.
            # The `target_seq_length` is just a rough target however, whereas
            # `max_seq_length` is a hard limit.
            # 10% of the time: have shorter sequences
            target_seq_length = random.randint(2, max_num_tokens) if (
                    random.random() < 0.1) else max_num_tokens

            # We DON'T just concatenate all of the tokens from a document into a long
            # sequence and choose an arbitrary split point because this would make the
            # next sentence prediction task too easy. Instead, we split the input into
            # segments "A" and "B" based on the actual "sentences" provided by the user
            # input.
            current_chunk = []
            current_length = 0
            for document in all_documents:
                i = 0
                while i < len(document):
                    segment = document[i]
                    if len(segment) > 0:
                        current_chunk.append(segment)
                    current_length += len(segment)
                    if i == len(document) - 1 or current_length >= target_seq_length:
                        if current_chunk:
                            # `a_end` is how many segments from `current_chunk` go into the `A`
                            # (first) sentence.
                            a_end = 1
                            if len(current_chunk) >= 2:
                                a_end = random.randint(1, max(1, len(current_chunk) - 1))

                            tokens_a = []
                            for j in range(a_end):
                                tokens_a.extend(current_chunk[j])

                            tokens_b = []
                            # Random next
                            # not_in_order = False
                            # Random next sentence not an option for me
                            if len(current_chunk) == 1:  # or (FLAGS.random_next_sentence and random.random() < 0.5):
                                # Case where there was only 1 sentence left for the current chunk:
                                # Use the previous segment (whether it was from the same document or not)
                                # switch them, and mark as 'not in order'
                                not_in_order = True

                                # Original code uses random chunk for not-in order b. I use previous token_b, and flip to be sure of not in order
                                # # This should rarely go for more than one iteration for large
                                # # corpora. However, just to be careful, we try to make sure that
                                # # the random document is not the same as the document
                                # # we're processing.
                                # target_b_length = target_seq_length - len(tokens_a)
                                # for _ in range(10):
                                #     random_document_index = rng.randint(0, len(all_documents) - 1)
                                #     if random_document_index != document_index:
                                #         break
                                #
                                # random_document = all_documents[random_document_index]
                                # random_start = rng.randint(0, len(random_document) - 1)
                                # for j in range(random_start, len(random_document)):
                                #     tokens_b.extend(random_document[j])
                                #     if len(tokens_b) >= target_b_length:
                                #         break
                                if previous_segment is not None:
                                    # previous_tensor = tensor_list[-1][-1]
                                    # sep_idxs = [i for i, el in enumerate(previous_tensor) if
                                    #             el == self.token_indexer.sep_token_id]
                                    # assert len(sep_idxs) == 2
                                    # # this b might be too big, but it will get truncated later anyway
                                    # tokens_b = previous_tensor[sep_idxs[0] + 1:sep_idxs[1]].tolist()

                                    tokens_a, tokens_b = previous_segment, tokens_a
                                else:
                                    # If it happens that we have a too-short document at the very start of this collection of documents, we just skip it
                                    log.info(f'One-sentence document without predecessor, in {path}: skipping.')
                                    i += 1
                                    continue

                                # We didn't actually use these segments so we "put them back" so
                                # they don't go to waste.
                                num_unused_segments = len(current_chunk) - a_end
                                i -= num_unused_segments
                            elif random.random() < 0.5:
                                not_in_order = True
                                for j in range(a_end, len(current_chunk)):
                                    tokens_b.extend(current_chunk[j])
                                # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                                tokens_a, tokens_b = tokens_b, tokens_a
                            # Actual next
                            else:
                                not_in_order = False
                                for j in range(a_end, len(current_chunk)):
                                    tokens_b.extend(current_chunk[j])
                            if (len(tokens_a) == 0) or (len(tokens_b) == 0):
                                print('breakpoint here')
                                raise
                            self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                            if (len(tokens_a) == 0) or (len(tokens_b) == 0):
                                print('breakpoint here')
                                raise
                            current_sequence = self.token_indexer.prepare_for_model(tokens_a, tokens_b,
                                                                                    truncation_strategy='do_not_truncate',
                                                                                    pad_to_max_length=True)['input_ids']
                            # tokens = []
                            # segment_ids = []
                            # tokens.append("[CLS]")
                            # segment_ids.append(0)
                            # for token in tokens_a:
                            #     tokens.append(token)
                            #     segment_ids.append(0)
                            #
                            # tokens.append("[SEP]")
                            # segment_ids.append(0)
                            #
                            # for token in tokens_b:
                            #     tokens.append(token)
                            #     segment_ids.append(1)
                            # tokens.append("[SEP]")
                            # segment_ids.append(1)
                            #
                            # (tokens, masked_lm_positions,
                            #  masked_lm_labels, token_boundary) = create_masked_lm_predictions(
                            #     tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                            # instance = TrainingInstance(
                            #     tokens=tokens,
                            #     segment_ids=segment_ids,
                            #     is_random_next=not_in_order,
                            #     token_boundary=token_boundary,
                            #     masked_lm_positions=masked_lm_positions,
                            #     masked_lm_labels=masked_lm_labels)
                            tensor_list.append(torch.tensor(current_sequence).unsqueeze(
                                0))
                            sentence_order_list.append(int(not_in_order))
                        current_chunk = []
                        current_length = 0
                    i += 1
                    if len(tensor_list) > 0:
                        assert len(tokens_b) > 0
                        previous_segment = tokens_b.copy()

        else:
            max_raw_seq_length = FLAGS.max_seq_length - 2  # Excluding [CLS] at start and [SEP] at end
            for token_ids in all_documents:
                for i in range(0, len(token_ids), max_raw_seq_length):
                    current_sequence = self.token_indexer.prepare_for_model(token_ids[i:i + max_raw_seq_length],
                                                                            truncation_strategy='do_not_truncate',
                                                                            pad_to_max_length=True)['input_ids']
                    tensor_list.append(torch.tensor(current_sequence).unsqueeze(
                        0))
        return {"ids": [torch.cat(tensor_list).to(torch.int32)],
                "order_label": sentence_order_list}

    def split_into_sentences(self, text):
        """
        Splits text into sentences if the sentence order prediction objective is used
        :param documents: a list of lists (documents) of token indices
        :return: a list of lists (documents) of lists (sentences) of token indices
        """
        if self.corpus == 'wiki':
            article_without_title = '\n\n'.join(text.split('\n\n')[1:])
            text = article_without_title
        return SENTENCE_TOKENIZER.tokenize(text)

        # sentence_separator_id = self.token_indexer.convert_tokens_to_ids(".")
        # result = []
        # for d in documents:
        #     idxs = [i+1 for i, el in enumerate(d) if el == sentence_separator_id]
        #     split_d = [d[i: j] for i, j in
        #                zip([0] + idxs, idxs +
        #                    ([len(d)] if idxs and (idxs[-1] != len(d)) else []))]
        #     result.append(split_d)
        # return result

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


class CombinedSplitDataset(IterableDataset):
    def __init__(self, split):
        super().__init__()
        self.token_indexer = get_my_tokenizer()
        self.split_name = split
        self.split_chunks_folder = Path(BLOB_SUBFOLDER, f'{split}')
        self.chunk_paths = None
        self.pop_indices = None
        self.row_index = None
        self.current_permuted_indices = None
        self.current_chunk_path = None

    def make_chunks(self):
        total_data_dict = self.get_data()
        split_into_chunks(self.split_name, total_data_dict)

    def __iter__(
            self):  # TODO make sure this supports multi-GPU loading with worker_info = torch.utils.data.get_worker_info(): https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        if not self.split_chunks_folder.is_dir():
            self.make_chunks()

        if (not self.current_chunk_path) and ((not self.chunk_paths) or (
                not self.pop_indices)):  # Storing this to be able to pick up runs intra-epoch between restarts
            self.chunk_paths = list(self.split_chunks_folder.glob('*.pt'))
            length = len(self.chunk_paths)
            self.pop_indices = []
            while length > 0:
                self.pop_indices.append(random.randrange(length))
                length -= 1
            assert self.chunk_paths, f"{self.split_chunks_folder} is empty!"

        while self.chunk_paths or self.current_chunk_path:
            assert len(self.pop_indices) == len(
                self.chunk_paths)  # Needs to be torch.long for downstream, but storing as int32 because uses less space
            should_get_new_chunk = ((self.current_chunk_path is None) or (self.current_permuted_indices is None))
            if should_get_new_chunk:
                pop_idx = self.pop_indices.pop(0)
                self.current_chunk_path = self.chunk_paths.pop(pop_idx)
            chunk_data = torch.load(self.current_chunk_path.as_posix())
            if BLOB_SUBFOLDER.name == "pairwise_segment_sequences":
                chunk_ids, chunk_order_label = chunk_data['ids'].to(torch.int64), chunk_data['order_label']
            else:
                chunk_ids = chunk_data
            if should_get_new_chunk:
                self.current_permuted_indices = torch.randperm(len(chunk_ids))

            chunk_ids = chunk_ids[self.current_permuted_indices]
            if BLOB_SUBFOLDER.name == "pairwise_segment_sequences":
                chunk_order_label = torch.tensor(chunk_order_label)
                chunk_order_label = chunk_order_label[self.current_permuted_indices]
            if not self.row_index:
                self.row_index = 0
            while self.row_index < len(chunk_ids):
                if BLOB_SUBFOLDER.name == "pairwise_segment_sequences":
                    yield {'input_ids': chunk_ids[self.row_index],
                           'sentence_order_labels': chunk_order_label[self.row_index]}
                else:
                    yield {'input_ids': chunk_ids[self.row_index]}

                self.row_index += 1
            self.current_permuted_indices = None
            self.current_chunk_path = None
            self.row_index = None

    def __len__(self):
        length_blob_path = Path(BLOB_SUBFOLDER, f'{self.split_name}_tensor_combined_length').as_posix()
        if os.path.exists(length_blob_path) and not FLAGS.fresh_data:
            length = torch.load(length_blob_path, map_location='cpu')
        else:
            length = len(self.get_data()['ids'])
            torch.save(length, length_blob_path)
        return int(length / FLAGS.d_batch)

    def get_data(self):
        blob_path = Path(BLOB_SUBFOLDER, f'{self.split_name}_tensor_combined').as_posix()
        if os.path.exists(blob_path) and not FLAGS.fresh_data:
            log.info(f'Loading {blob_path} ...')
            start = time.time()
            result = torch.load(blob_path, map_location='cpu')
            log.info(f'Loaded {blob_path} in {time.time() - start:.2} seconds')
        else:
            result = self.combine_data()
            torch.save(result, blob_path)
        return result

    def combine_data(self):
        corpus_names = ['wiki', 'gutenberg', 'bookcorpus']
        single_datasets_data = [SingleDataset(corpus_name=corpus_name, split_name=self.split_name).get_data()
                                for corpus_name in corpus_names]
        unflattened = {k: [dic[k] for dic in single_datasets_data] for k in single_datasets_data[0]}
        flattened = {
            'ids': torch.cat(unflattened['ids']),
            'order_label': [l for corpus_labels in unflattened['order_label'] for l in corpus_labels]
        }
        return flattened


def get_data_dict():
    '''
    Returns a dictionary containing train, test and validation instance lists, as well as the vocab created from train and validation data
    '''
    if not os.path.exists(BLOB_SUBFOLDER):
        os.makedirs(BLOB_SUBFOLDER)

    train_dataset, test_dataset, val_dataset = [
        CombinedSplitDataset(f'{FLAGS.pretrain_data_fraction}_{split}_{FLAGS.max_seq_length}')
        for split in ['train', 'test', 'val']]
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
            result = torch.load(self.blob_path, map_location='cpu')
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
    if not os.path.exists(BLOB_SUBFOLDER):
        os.makedirs(BLOB_SUBFOLDER)
    train_dataset, test_dataset, val_dataset = [
        GutenbergSplitDataset(Path(FLAGS.pretrain_data_folder, 'Gutenberg', split).as_posix(),
                              Path(BLOB_SUBFOLDER, f'{split}_tensor_{FLAGS.max_seq_length}').as_posix())
        for split in ['train', 'test', 'val']]

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


def split_into_chunks(split_name, split_dict):
    split_tensor = split_dict['ids']
    split_order_labels = split_dict['order_label']
    chunk_size_MiB = 500  # Size of chunks to load into memory at once, in MiB
    B_per_el = split_tensor.element_size()
    nb_cols = split_tensor.shape[-1]
    B_per_MiB = 2 ** 20
    els_per_chunk = ((chunk_size_MiB * B_per_MiB) / B_per_el)
    rows_per_chunk = int(els_per_chunk / nb_cols)
    split_dir = Path(BLOB_SUBFOLDER, split_name)
    Path.mkdir(split_dir)
    for i in range(0, len(split_tensor), rows_per_chunk):
        ids_chunk = split_tensor[i:i + rows_per_chunk].clone()
        order_labels_chunk = split_order_labels[i:i + rows_per_chunk].copy()
        path = Path(split_dir, f'{i}.pt').as_posix()
        torch.save({'ids': ids_chunk, 'order_label': order_labels_chunk}, path)
        del ids_chunk, order_labels_chunk
