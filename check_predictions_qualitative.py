import torch
from main import GutenbergReader
import os
from config import FLAGS
from allennlp.data import Vocabulary
from model import FullModel
from pathlib import Path

reader = GutenbergReader()
train_dataset = reader.read(os.path.join(FLAGS.data_folder,'train'))
test_dataset = reader.read(os.path.join(FLAGS.data_folder,'test'))
val_dataset = reader.read(os.path.join(FLAGS.data_folder,'val'))

vocab = Vocabulary.from_instances(train_dataset + val_dataset)

vocab.add_token_to_namespace("@@PADDING@@")


model = FullModel(vocab)
trained_model_path = Path('.','output','dummy_baseline','best.th')
model.load_state_dict(torch.load(trained_model_path))
cuda_device = FLAGS.device_idx
model = model.cuda(cuda_device)

sample_idx = 0
test_dataset[sample_idx].index_fields(vocab)
og_indices = list((a._indexed_tokens for a in test_dataset[sample_idx].fields.values()))
og_indices_tensor = [{key:torch.tensor(value).cuda(cuda_device)} for dict in og_indices for (key,value) in dict.items()]
model(*og_indices_tensor)
print(5)