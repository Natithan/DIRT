import os
os.environ['TORCH_HOME'] = os.path.join('/cw', 'working-arwen', 'nathan')
os.environ['ALLENNLP_CACHE_ROOT'] = os.path.join('/cw', 'working-arwen', 'nathan')

import random

from wrappers import MLMModelWrapper, MODEL_MAPPING

import sys

import torch
from absl import app

from config import FLAGS
from model import FullModel
from pathlib import Path

from text_input_pipeline import GutenbergReader

def main(_):
    reader = GutenbergReader()
    train_dataset, test_dataset, val_dataset, vocab = (reader.get_data_dict()[key] for key in ('train','test','val','vocab'))
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model],vocab)

    trained_model_path = Path(latest_run_dir, 'best.th')
    model.load_state_dict(torch.load(trained_model_path))
    cuda_device = FLAGS.device_idx
    model = model.cuda(cuda_device)
    model.eval() # Set to eval mode
    for name,dataset in zip(('Train','Test','Val'),(train_dataset,test_dataset,val_dataset)):
        print(f'Testing {name}')
        instances = random.sample(dataset,3)

        inputs = [[instance.fields['inputs'].tokens] for instance in instances]
        prediction = model.forward_on_instances(instances) # TODO make sure this is up-to-date with current model wrapper. Make sure it also outputs results instead of only loss
        predicted_words = [[vocab.get_token_from_index(i) for i in sequence['prediction']] for sequence in prediction] #TODO avoid my model just outputting 'mask' all of the time ;)
        # actual_target = #TODO Avoid HF model to always output mask tokens :P
        for input, target, prediction in zip(inputs, targets, predicted_words):
            print(f'Input: {input}')
            print(f'Target: {target}')
            print(f'Prediction: {prediction}')
            print(' ')

if __name__ == '__main__':
    all_runs = [Path('.','output', model_dir, run)
                for model_dir in os.listdir(Path('.', 'output')) for run in os.listdir(Path('.', 'output', model_dir))
                if os.path.isdir(Path('.', 'output', model_dir)) and os.path.isdir(Path('.', 'output', run))]
    latest_run_dir = max(all_runs, key=os.path.getmtime)
    flagfile = Path(latest_run_dir, 'flagfile.txt')
    app.run(main,sys.argv + [f'--flagfile={flagfile}'])