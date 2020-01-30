import random
import os
import torch
from config import FLAGS
from model import FullModel
from pathlib import Path

from text_input_pipeline import GutenbergReader

reader = GutenbergReader()  # TODO add COPA task later
train_dataset, test_dataset, val_dataset, vocab = (reader.get_data_dict()[key] for key in ('train','test','val','vocab'))
model = FullModel(vocab)

all_runs = [Path('.','output',d) for d in os.listdir(Path('.','output')) if os.path.isdir(Path('.','output',d))]
latest_run_dir = max(all_runs, key=os.path.getmtime)
trained_model_path = Path(latest_run_dir, 'best.th')
model.load_state_dict(torch.load(trained_model_path))
cuda_device = FLAGS.device_idx
model = model.cuda(cuda_device)
model.eval() # Set to eval mode
instances = random.sample(test_dataset,3)

inputs = [[instance.fields['inputs'].tokens] for instance in instances]
targets = [[instance.fields['targets'].tokens] for instance in instances]
prediction = model.forward_on_instances(random.sample(test_dataset,3))
predicted_words = [[vocab.get_token_from_index(i) for i in sequence['prediction']] for sequence in prediction] #TODO avoid the model just outputting 'mask' all of the time ;)
# actual_target =
for input, target, prediction in zip(inputs, targets, predicted_words):
    print(f'Input: {input}')
    print(f'Target: {target}')
    print(f'Prediction: {prediction}')
    print(' ')