from copy import deepcopy
import itertools
import random

import torch
from allennlp.data import Token

from config import FLAGS
from constants import MASKING_TOKEN, TO_BE_DELETED_TOKEN, EOS_TOKEN, BPE_INDEXER_SUFFIX


def t5_denoise_spans_objective(
        tokens):  # Based on objective in t5 paper: https://arxiv.org/abs/1910.10683 #TODO update this to work with tensors instead of lists
    '''
    Produces inputs and targets.
    Inputs correspond to the original tokens, with a certain fraction of tokens replaced by a MASK-token.
    Contiguous tokens that happen to get masked get replaced by a single MASK token.
    There is no switching with random words, ... ( “MASS-style” objective )
    Targets look like: [mask_0, *<first word that was masked, possibly multiple if contiguous>, mask_1, <same>, ... , mask_eos, padding, padding, ... >
    '''
    masked_indices = sorted(random.sample(range(len(tokens)), int(len(tokens) * FLAGS.masking_fraction)))  #

    given = [t if (i not in masked_indices) else MASKING_TOKEN for i, t in enumerate(tokens)]
    masked_given = [i for i, j in zip(given[1:], given[:-1]) if not (i == MASKING_TOKEN and i == j)]
    mask_counter = itertools.count()
    unique_masked_given = [Token(f'{i}_{next(mask_counter)}') if i == MASKING_TOKEN else i for i in masked_given]

    target = [tokens[i] for i in masked_indices]
    include_mask = [True] + [((i - j) != 1) for i, j in zip(masked_indices[1:], masked_indices[:-1])]
    masks = [MASKING_TOKEN if x else TO_BE_DELETED_TOKEN for x in include_mask]
    masked_target = [i for j in zip(masks, target) for i in j if i != TO_BE_DELETED_TOKEN]
    mask_counter = itertools.count()  # Restart count
    unique_masked_target = [Token(f'{i}_{next(mask_counter)}') if i == MASKING_TOKEN else i for i in masked_target] + [
        Token(EOS_TOKEN)]
    return unique_masked_given, unique_masked_target


def simple_mlm_objective(input_ids, token_indexer):
    '''
    Produces a tensor of the same shape as masked_lm_labels, but with FLAGS.masking_fraction of the tokens replaces by a mask id
    '''
    masking_id = token_indexer.mask_token_id
    not_special_token = deepcopy(input_ids).cpu().apply_(lambda x: x not in token_indexer.all_special_ids).cuda(
        input_ids.device).to(torch.bool)
    is_target_idx = (torch.rand(input_ids.shape).cuda(input_ids.device) <= FLAGS.masking_fraction) & \
                    not_special_token
    masked_ids = torch.where(is_target_idx,
                             masking_id * torch.ones_like(input_ids),
                             input_ids)
    return masked_ids, is_target_idx


# TODO add albert-style objective: BERT-like MLM + SOP
def BERT_MLM_objective(input_ids, token_indexer):
    '''
    Produces a tensor of the same shape as masked_lm_labels, but with
    - FLAGS.masking_fraction of the tokens replaced by a mask id
    - FLAGS.preserve_fraction * FLAGS.masking_fraction
    - FLAGS.random_switch_fraction
    '''
    masked_ids, is_target_idx = simple_mlm_objective(input_ids, token_indexer)
    random_tensor = torch.rand(input_ids.shape).cuda(input_ids.device)
    random_ids = torch.randint(max(token_indexer.all_special_ids) + 1, token_indexer.vocab_size, input_ids.shape).to(input_ids.device)
    predict_original = random_tensor < FLAGS.preserve_fraction
    predict_random = (FLAGS.preserve_fraction < random_tensor) & \
                     (random_tensor < (FLAGS.random_switch_fraction + FLAGS.preserve_fraction))
    mixed_masked_ids = torch.where(predict_original & is_target_idx,
                                   input_ids,
                                   torch.where(predict_random & is_target_idx,
                                               random_ids,
                                               masked_ids))
    return mixed_masked_ids, is_target_idx
