from torch import nn

from transformers import RobertaForMaskedLM


class RobertaMLMWrapper(nn.Module):
    '''
    Wrapper class for huggingface's RobertaForMaskedLM to allow passing it to the AllenNLP trainer
    '''
    def __init__(self,model):
        assert isinstance(model,RobertaForMaskedLM)
        self.model = model
    def forward(self, input_dict, target_dict):
        input_ids, target_ids = input_dict['ids'], (target_dict['ids'] if (target_dict is not None) else None)
        tuple_result = self.model(input_ids=input_ids,masked_lm_labels=target_ids)
        result_dict = {}
        if target_ids is not None:
            result_dict['loss'] = tuple_result[0] #Add more parts of output when needed :P
        return result_dict
