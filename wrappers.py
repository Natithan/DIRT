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

    def forward(self, inputs): #for now ignore ids-offsets and word-level padding mask: just use bpe-level tokens
        new_input_dict = {}
        new_input_dict['target_ids'] = inputs['ids']
        new_input_dict['padding_mask'] = inputs['ids'] != 0
        new_input_dict['masked_ids'] = self.objective(inputs['ids'],self.vocab)
        return self.model(**new_input_dict)


class RobertaMLMWrapper(Model):
    '''
    Wrapper class for huggingface's RobertaForMaskedLM to allow passing it to the AllenNLP trainer
    '''

    def __init__(self, dummy_vocab):
        super().__init__(dummy_vocab)
        config_path = CONFIG_MAPPING[FLAGS.model]
        model_class = RobertaForMaskedLM
        config = model_class.config_class.from_pretrained(config_path) #TODO find out if I'm using a pretrained model that is trained on different ids for words
        self.model = model_class(config)

    def forward(self, target_ids, masked_ids, padding_mask):
        tuple_result = self.model(input_ids=masked_ids, masked_lm_labels=target_ids,attention_mask = padding_mask)
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = tuple_result[0]  # Add more parts of output when needed :P
            result_dict['vocab_logits'] = tuple_result[1]
        else:
            result_dict['vocab_logits'] = tuple_result[0]
        return result_dict

class RobertaTokenizerWrapper(TokenIndexer): #TODO make sure I tokenize with useful indices if using pretrained model

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