import itertools
import random

from allennlp.data import Token

from config import FLAGS
from constants import MASKING_TOKEN, TO_BE_DELETED_TOKEN, EOS_TOKEN


def t5_denoise_spans_objective(tokens):  # Based on objective in t5 paper: https://arxiv.org/abs/1910.10683
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


def BERT_MLM_objective(tokens): #TODO maybe add dynamic masking? I.e. at every entrance to the model
    '''
    Produces inputs and targets.
    Inputs correspond to the original tokens, with a certain fraction of tokens replaced by a MASK-token.
    There is no switching with random words, ... ( “MASS-style” objective )
    Targets are the uncorrupted tokens
    '''
    masked_indices = sorted(random.sample(range(len(tokens)), int(len(tokens) * FLAGS.masking_fraction)))  #

    # inputs = [Token(t) if (i not in masked_indices) else Token(MASKING_TOKEN) for i, t in enumerate(tokens)]
    # targets = [Token(t) for t in tokens]
    inputs = [t if (i not in masked_indices) else Token(MASKING_TOKEN) for i, t in enumerate(tokens)]
    targets = [t for t in tokens]
    return inputs, targets
    #TODO FINISH THIS HERE BELOW
    bpe_mask_token = self.token_indexers['ids'].byte_pair_encode(Token(MASKING_TOKEN))[0]
    masking_id = vocab.get_token_index(bpe_mask_token,'openai_transformer')
    input_ids = instance.as_tensor_dict()['inputs']['ids']
    masked_input_ids = torch.where(torch.rand_like(input_ids) < FLAGS.masking_fraction, input_ids, masking_id * torch.ones_like(input_ids).to(torch.int32))
