from collections import OrderedDict

from allennlp.models import Model
from typing import Dict, List

from allennlp.data import TokenIndexer, TokenType, Token, Vocabulary
from torch import nn

from config import FLAGS, CONFIG_MAPPING, OBJECTIVE_MAPPING
from model import FullModel
from transformers import RobertaForMaskedLM, RobertaTokenizer

class MLMModelWrapper(Model):
    def __init__(self,model,vocab):
        super().__init__(vocab)
        self.model = model(vocab)
        self.objective = OBJECTIVE_MAPPING[FLAGS.objective]

    def forward(self, inputs):
        new_input_dict = {}
        new_input_dict['target_ids'] = inputs['ids']
        new_input_dict['mask'] = inputs['mask']
        new_input_dict['masked_ids'] = self.objective(inputs['ids'],self.vocab)
        result_dict = self.model(**new_input_dict) #TODO fix TypeError: forward() got an unexpected keyword argument 'mask'
        return result_dict


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

    def forward(self, target_ids, masked_ids):
        input_ids, input_padding_mask = target_ids['ids'], target_ids['mask']
        target_ids, target_padding_mask = (masked_ids['ids'], masked_ids['mask']) if (masked_ids is not None) else (None, None)
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


MODEL_MAPPING = OrderedDict(
    [
        ("huggingface_baseline_encoder", RobertaMLMWrapper,),
        ("my_baseline_encoder", FullModel,),
    ]
)