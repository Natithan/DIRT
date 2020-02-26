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

    best_val_run_path = Path(latest_run_dir, 'best.th')
    model_states = [Path(latest_run_dir,m) for m in os.listdir(latest_run_dir) if 'model_state_epoch_' in m]
    latest_run_path = max(model_states, key=os.path.getmtime)

    for trained_model_path in (best_val_run_path,latest_run_path):
        print(f'Testing {trained_model_path}')
        model.load_state_dict(torch.load(trained_model_path))
        cuda_device = FLAGS.device_idx
        model = model.cuda(cuda_device)
        model.eval() # Set to eval mode
        for name,dataset in zip(('Train','Test','Val'),(train_dataset,test_dataset,val_dataset)):
            print(f'Testing {name}')
            instances = random.sample(dataset,3)

            inputs = [[instance.fields['inputs'].tokens] for instance in instances]
            prediction = model.forward_on_instances(instances)
            predicted_words = [[vocab.get_token_from_index(index,'openai_transformer') for index in prediction[batch_sample]['vocab_logits'].argmax(1)] for batch_sample in range(len(prediction))] #TODO avoid my model just outputting 'mask' all of the time ;)
            for input, prediction in zip(inputs, predicted_words):
                print(f'Input: {input}')
                print(f'Prediction: {prediction}')
                print(' ')

if __name__ == '__main__':
    all_runs = [Path('.','output', model_dir, run)
                for model_dir in os.listdir(Path('.', 'output')) for run in os.listdir(Path('.', 'output', model_dir))
                if os.path.isdir(Path('.', 'output', model_dir)) and os.path.isdir(Path('.', 'output', run))]
    latest_run_dir = max(all_runs, key=os.path.getmtime)
    flagfile = Path(latest_run_dir, 'flagfile.txt')
    app.run(main,sys.argv + [f'--flagfile={flagfile}'])