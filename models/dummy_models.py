from collections import Counter
from typing import Dict

import torch
from allennlp.models import Model
from torch import nn

from config import FLAGS, TOKENIZER_MAPPING


class RandomMLMModel(Model):
    """
    Model that picks word at the masked indices at random.
    Used to put perplexity in perspective
    """

    def __init__(self, vocab):
        super().__init__(vocab)
        self.dummy_param = torch.nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, target_ids, masked_ids, padding_mask) -> Dict[str, torch.Tensor]:
        tokenizer = TOKENIZER_MAPPING[FLAGS.model]
        vocab_scores = torch.rand(target_ids.shape[0], target_ids.shape[1], tokenizer.vocab_size).cuda()
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = nn.CrossEntropyLoss()(vocab_scores.contiguous().view(-1, tokenizer.vocab_size),
                                                        target_ids.contiguous().view(-1)) \
                                  + self.dummy_param - self.dummy_param  # To trick the trainer ;)
        result_dict['vocab_scores'] = vocab_scores
        return result_dict


class ConstantMLMModel(Model):
    """
    Model that always picks the most common word in the training data to replace masked tokens
    Used to put perplexity in perspective
    """

    def __init__(self, vocab):
        super().__init__(vocab)
        self.dummy_param = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.token_counts = Counter()

    def forward(self, target_ids, masked_ids, padding_mask) -> Dict[str, torch.Tensor]:
        tokenizer = TOKENIZER_MAPPING[FLAGS.model]
        float_masked_ids = masked_ids.to(torch.float).clone()
        # Replace mask-ids with random floats to make sure they are not the most common element
        maskless_masked_ids = torch.where(masked_ids == tokenizer.mask_token_id, torch.rand_like(float_masked_ids), float_masked_ids)
        # Pick the most common element in each sample
        most_common_ids = maskless_masked_ids.mode()[0].to(torch.long)
        vocab_scores = torch.zeros(target_ids.shape[0], target_ids.shape[1], tokenizer.vocab_size).cuda()

        mask_idxs = (masked_ids == tokenizer.mask_token_id).nonzero()
        for batch_idx, common_id in enumerate(most_common_ids.tolist()): #TODO maybe change loss calculation to only consider masked positions
            single_sample_mask_idxs = mask_idxs[(mask_idxs[:, 0] == batch_idx).nonzero().squeeze(1)]
            for sequence_idx in range(masked_ids.shape[1]):
                if sequence_idx in single_sample_mask_idxs:
                    vocab_scores[batch_idx, sequence_idx, common_id] = 1 #TODO check if this works
                else:
                    existing_id = masked_ids[batch_idx,sequence_idx]
                    vocab_scores[batch_idx, sequence_idx, existing_id] = 1
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = nn.CrossEntropyLoss()(vocab_scores.contiguous().view(-1, tokenizer.vocab_size),
                                                        target_ids.contiguous().view(-1))\
                                  + self.dummy_param - self.dummy_param
        result_dict['vocab_scores'] = vocab_scores
        return result_dict
