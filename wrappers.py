from allennlp.models import Model
from typing import Dict, List

from allennlp.data import TokenIndexer, TokenType, Token, Vocabulary
from torch import nn

from config import FLAGS, CONFIG_MAPPING
from transformers import RobertaForMaskedLM, RobertaTokenizer


class RobertaMLMWrapper(Model):
    '''
    Wrapper class for huggingface's RobertaForMaskedLM to allow passing it to the AllenNLP trainer
    '''

    def __init__(self, dummy_vocab):
        super().__init__(dummy_vocab)
        config_path = CONFIG_MAPPING[FLAGS.model]
        model_class = RobertaForMaskedLM
        config = model_class.config_class.from_pretrained(config_path)
        self.model = model_class(config)

    def forward(self, inputs, targets):
        input_ids, input_padding_mask = inputs['ids'], inputs['mask']
        target_ids, target_padding_mask = (targets['ids'], targets['mask']) if (targets is not None) else (None, None)
        tuple_result = self.model(input_ids=input_ids, masked_lm_labels=target_ids,attention_mask = input_padding_mask)
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = tuple_result[0]  # Add more parts of output when needed :P
        return result_dict


class RobertaTokenizerWrapper(TokenIndexer):

    def __init__(self, tokenizer, namespace='tokens'):
        super().__init__()
        assert isinstance(tokenizer, RobertaTokenizer)
        self.tokenizer = tokenizer
        self.namespace = namespace

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        text = token.text
        counter[self.namespace][text] += 1

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary, index_name: str) -> Dict[
        str, List[TokenType]]:
        pass

    def get_padding_lengths(self, token: TokenType) -> Dict[str, int]:
        pass

    def pad_token_sequence(self, tokens: Dict[str, List[TokenType]], desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, TokenType]:
        pass
