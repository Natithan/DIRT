from collections import Counter
from typing import Dict

import torch
from allennlp.models import Model
from torch import nn

from config import FLAGS

from models.wrappers import TOKENIZER_MAPPING


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
        vocab_scores = torch.rand(target_ids.shape[0], target_ids.shape[1], tokenizer.vocab_size).cuda(FLAGS.device_idx)
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = nn.CrossEntropyLoss()(vocab_scores.view(-1, tokenizer.vocab_size),
                                                        target_ids.view(-1)) \
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
        tokenizer = TOKENIZER_MAPPING[FLAGS.model] #TODO finish this
        vocab_scores = torch.rand(target_ids.shape[0], target_ids.shape[1], tokenizer.vocab_size).cuda(
            FLAGS.device_idx)
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = nn.CrossEntropyLoss()(vocab_scores.view(-1, tokenizer.vocab_size),
                                                        target_ids.view(-1))
        result_dict['vocab_scores'] = vocab_scores
        return result_dict
