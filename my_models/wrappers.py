from collections import OrderedDict

import torch
from allennlp.models import Model
from typing import Dict, List

from allennlp.data import TokenIndexer, Token, Vocabulary

from config import FLAGS, CONFIG_MAPPING, OBJECTIVE_MAPPING, TOKENIZER_MAPPING
from transformers import RobertaForMaskedLM, RobertaTokenizer


class MLMModelWrapper(Model):
    def __init__(self, model):
        super().__init__(Vocabulary())
        self.model = model()
        self.objective = OBJECTIVE_MAPPING[FLAGS.objective]
        self.token_indexer = TOKENIZER_MAPPING[FLAGS.model]

    def forward(self, input_ids):  # for now ignore ids-offsets and word-level padding mask: just use bpe-level tokens
        new_input_dict = {}
        new_input_dict['padding_mask'] = input_ids != self.token_indexer.pad_token_id
        new_input_dict['masked_ids'] = self.objective(input_ids, self.token_indexer)
        new_input_dict['masked_lm_labels'] = torch.where(
            new_input_dict['masked_ids'] == self.token_indexer.mask_token_id, input_ids,
            torch.ones_like(input_ids) * (-100))
        result_dict = self.model(**new_input_dict)
        result_dict['mask'] = (new_input_dict['masked_ids'] == self.token_indexer.mask_token_id)
        return result_dict

    def get_metrics(self, **kwargs):
        return self.model.get_metrics()


class RobertaMLMWrapper(Model):
    '''
    Wrapper class for huggingface's RobertaForMaskedLM to allow passing it to the AllenNLP trainer
    '''

    def __init__(self, dummy_vocab):
        super().__init__(dummy_vocab)
        self.metrics_dict = {}
        config_name = CONFIG_MAPPING[FLAGS.model]
        model_class = RobertaForMaskedLM
        if FLAGS.use_pretrained_weights:
            self.model = model_class.from_pretrained(config_name)
        else:
            config = model_class.config_class.from_pretrained(
                config_name)
            self.model = model_class(config)

    def forward(self, masked_lm_labels, masked_ids, padding_mask):
        tuple_result = self.model(input_ids=masked_ids, masked_lm_labels=masked_lm_labels, attention_mask=padding_mask)
        result_dict = {}
        if masked_lm_labels is not None:
            result_dict['loss'] = tuple_result[0]  # Add more parts of output when needed :P
            self.metrics_dict['crossentropy_loss'] = result_dict['loss'] #TODO figure out why crossentropy loss much more jittery than normal loss
            result_dict['vocab_scores'] = tuple_result[1]
        else:
            result_dict['vocab_scores'] = tuple_result[0]
        return result_dict

    def get_metrics(self, **kwargs):
        return self.metrics_dict


class RobertaTokenizerWrapper(TokenIndexer):

    def __init__(self, namespace='tokens'):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(CONFIG_MAPPING[
                                                              'huggingface_baseline_encoder'])  # TODO maybe refactor later to deal with non-from-pretrained model
        self.namespace = namespace

    def encode(self, text):
        return self.tokenizer.encode(text)

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        text = token.text
        counter[self.namespace][text] += 1

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary, index_name: str):
        return self.tokenizer.encode(tokens, add_special_tokens=True)

    def get_padding_lengths(self, token) -> Dict[str, int]:
        pass

    def pad_token_sequence(self, tokens, desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]):
        pass


from my_models.dummy_models import RandomMLMModel, ConstantMLMModel

from my_models.model import DIRT

MODEL_MAPPING = OrderedDict(
    [
        ("hf_baseline", RobertaMLMWrapper,),
        ("my_model", DIRT,),
        ("random", RandomMLMModel,),
        ("constant", ConstantMLMModel,),
    ]
)