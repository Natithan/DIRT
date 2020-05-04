from collections import OrderedDict

import torch
from allennlp.models import Model
from typing import Dict, List

from allennlp.data import TokenIndexer, Token, Vocabulary

from config import FLAGS, OBJECTIVE_MAPPING, get_my_tokenizer
from transformers import AlbertForMaskedLM


class MLMModelWrapper(Model):
    def __init__(self, model,finetune_stage=False):
        super().__init__(Vocabulary())
        self.finetune_stage=finetune_stage
        self.model = model(finetune_stage)
        self.objective = OBJECTIVE_MAPPING[FLAGS.objective]
        self.token_indexer = get_my_tokenizer()

    def forward(self, input_ids,token_type_ids=None):  # for now ignore ids-offsets and word-level padding mask: just use bpe-level tokens
        new_input_dict = {}
        new_input_dict['padding_mask'] = input_ids != self.token_indexer.pad_token_id
        if (not self.finetune_stage):
            masked_ids = self.objective(input_ids, self.token_indexer)
            new_input_dict['input_ids'] = masked_ids
        else:
            new_input_dict['input_ids'] = input_ids
        new_input_dict['masked_lm_labels'] = torch.where(
            new_input_dict['input_ids'] == self.token_indexer.mask_token_id, input_ids,
            torch.ones_like(input_ids) * (-100))
        new_input_dict['token_type_ids'] = token_type_ids
        result_dict = self.model(**new_input_dict)
        result_dict['mask'] = (new_input_dict['input_ids'] == self.token_indexer.mask_token_id)
        return result_dict

    def get_metrics(self, **kwargs):
        return self.model.get_metrics()




class AlbertMLMWrapper(Model): #TODO change this to be AlbertWrapper
    '''
    Wrapper class for huggingface's AlbertForMaskedLM to allow passing it to the AllenNLP trainer
    '''

    def __init__(self, dummy_vocab,finetune_stage=False): #TODO if ever reuse this: adapt to finetune stage
        super().__init__(dummy_vocab)
        self.metrics_dict = {}
        model_class = AlbertForMaskedLM
        config_name = FLAGS.hf_model_handle
        if FLAGS.use_pretrained_weights:
            self.model = model_class.from_pretrained(config_name,output_hidden_states=True)
        else:
            config = model_class.config_class.from_pretrained(
                config_name,output_hidden_states=True)
            self.model = model_class(config)

    def forward(self,input_ids, padding_mask, masked_lm_labels=None, token_type_ids=None):
        tuple_result = self.model(input_ids=input_ids, masked_lm_labels=masked_lm_labels, attention_mask=padding_mask,token_type_ids=token_type_ids)
        result_dict = {}
        if masked_lm_labels is not None:
            result_dict['loss'] = tuple_result[0]  # Add more parts of output when needed :P
            self.metrics_dict['crossentropy_loss'] = result_dict['loss']
            self.metrics_dict['perplexity'] = torch.exp(result_dict['loss']).item()
            result_dict['vocab_scores'] = tuple_result[1]
        else:
            result_dict['vocab_scores'] = tuple_result[0]
        all_hidden_outputs = tuple_result[-1]
        result_dict['encoded_activations'] = all_hidden_outputs[-1] #TODO this is incorrect: AlbertForMaskedMLM doesn't give access to last-layer hidden states, and I need those for SG
        return result_dict

    def get_metrics(self, **kwargs):
        return self.metrics_dict.copy()


from dummy_models import RandomMLMModel, ConstantMLMModel

from model import DIRTLMHead

MODEL_MAPPING = OrderedDict(
    [
        ("hf_baseline", AlbertMLMWrapper,),
        ("my_model", DIRTLMHead,),
        ("random", RandomMLMModel,),
        ("constant", ConstantMLMModel,),
    ]
)