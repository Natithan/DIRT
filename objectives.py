import copy
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


def BERT_MLM_objective(target_ids, token_indexer):
    '''
    Produces a tensor of the same shape as masked_lm_labels, but with FLAGS.masking_fraction of the tokens replaces by a mask id
    '''
    masking_id = token_indexer.mask_token_id
    condition = (torch.rand(target_ids.shape).cuda(target_ids.device) > FLAGS.masking_fraction) | \
                (target_ids.cpu().apply_(lambda x: x in token_indexer.all_special_ids).cuda(target_ids.device).to(torch.bool))
    masked_ids = torch.where(condition,
                             target_ids,
                             masking_id * torch.ones_like(target_ids))
    return masked_ids
