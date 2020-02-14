# import torch
# from transformers import T5Tokenizer, T5WithLMHeadModel
#
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5WithLMHeadModel.from_pretrained('t5-small')
# input_ids = torch.tensor(tokenizer.encode("Hello my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids=input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
