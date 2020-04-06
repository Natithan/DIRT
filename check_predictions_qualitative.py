import os
import random

from my_models.wrappers import MLMModelWrapper, MODEL_MAPPING

import sys

import torch
from absl import app
import numpy as np

from config import FLAGS
from pathlib import Path

from text_input_pipeline import GutenbergReader
from my_utils.util import get_gpus_with_enough_memory

FIXED_DEVICE_IDXS = None #[0]
# CHOSEN_RUN_DIR = Path('output','constant','same')
CHOSEN_RUN_DIR = Path('output','huggingface_baseline_encoder','no_pretrain')

#TODO bring this up-to-date if gonna use
def main(_):
    FLAGS.device_idxs = get_gpus_with_enough_memory(
        8000) if not FIXED_DEVICE_IDXS else FIXED_DEVICE_IDXS  # Hack to not use flagvalue of og model when pertaining to GPU usage
    reader = GutenbergReader()
    data_dict = reader.get_data_dict()
    train_dataset, test_dataset, val_dataset, vocab = (data_dict[key] for key in
                                                       ('train', 'test', 'val', 'vocab'))
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model], vocab)
    model_device = f'cuda:{FLAGS.device_idxs[0]}' if len(FLAGS.device_idxs) != 0 else 'cpu'
    model = model.cuda(model_device)

    best_val_run_path = Path(latest_run_dir, 'best.th')
    model_states = [Path(latest_run_dir, m) for m in os.listdir(latest_run_dir) if 'model_state_epoch_' in m]
    latest_run_path = max(model_states, key=os.path.getmtime)

    for trained_model_path in (best_val_run_path, latest_run_path):
        print(f'Testing {trained_model_path}')
        model.load_state_dict(torch.load(trained_model_path,map_location=torch.device(model_device)))
        model = model.cuda(model_device)
        model.eval()  # Set to eval mode
        for name, dataset in zip(('Train', 'Test', 'Val'), (train_dataset, test_dataset, val_dataset)):
            print(f'Testing {name}')
            instances = random.sample(dataset, 1)
            predictions = model.forward_on_instances(
                instances)
            input_texts = [tokens_to_mask_aware_text(model.token_indexer,
                                                     model.token_indexer.convert_ids_to_tokens(
                                                         instances[batch_sample].fields['input_ids'].array),
                                                     predictions[batch_sample]['mask'])
                           for batch_sample in range(len(predictions))]
            input_ids = torch.cat([instance.as_tensor_dict()['input_ids'][None,:] for instance in instances]).numpy()
            prediction_ids = np.vstack([pred['vocab_scores'].argmax(1)[None,:] for pred in predictions])
            mask = np.vstack([predictions[batch_sample]['mask'] for batch_sample in range(len(predictions))])
            filled_in_predictions = np.where(mask,
                                               prediction_ids,
                                               input_ids)
            predicted_texts = [tokens_to_mask_aware_text(model.token_indexer,
                                                         model.token_indexer.convert_ids_to_tokens(
                                                             predictions[batch_sample]['vocab_scores'].argmax(1)),
                                                         predictions[batch_sample]['mask'])
                               for batch_sample in range(len(predictions))]

            filled_in_predicted_texts = [tokens_to_mask_aware_text(model.token_indexer,
                                                         model.token_indexer.convert_ids_to_tokens(
                                                             filled_in_predictions[batch_sample]),
                                                         predictions[batch_sample]['mask'])
                               for batch_sample in range(len(predictions))]
            for input_text, predicted_text, filled_in_predicted_text in zip(input_texts, predicted_texts, filled_in_predicted_texts):
                print(f'Input: {input_text}')
                print(f'Filled in Prediction: {filled_in_predicted_text}')
                print(f'All output: {predicted_text}')
                print(' ')


# Copied from HF transformers
def tokens_to_mask_aware_text(tokenizer, filtered_tokens, mask,
                              skip_special_tokens=False, clean_up_tokenization_spaces=True):
    # To avoid mixing byte-level and unicode for byte-level BPT
    # we need to build string separately for added tokens and byte-level tokens
    # cf. https://github.com/huggingface/transformers/issues/1133
    sub_texts = []
    current_sub_text = []
    for token, isMasked in zip(filtered_tokens, mask):
        if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
        if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
                sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))
                current_sub_text = []
            sub_texts.append(token)
        else:
            if isMasked:
                current_sub_text.append(tokenizer.mask_token)
                current_sub_text.append(token)
                current_sub_text.append(tokenizer.mask_token)
            else:
                current_sub_text.append(token)
    if current_sub_text:
        sub_texts.append(tokenizer.convert_tokens_to_string(
            current_sub_text))
    text = " ".join(sub_texts)

    if clean_up_tokenization_spaces:
        clean_text = tokenizer.clean_up_tokenization(text)
        return clean_text
    else:
        return text


if __name__ == '__main__':
    all_runs = [Path('.', 'output', model_dir, run)
                for model_dir in os.listdir(Path('.', 'output')) for run in os.listdir(Path('.', 'output', model_dir))
                if os.path.isdir(Path('.', 'output', model_dir)) and os.path.isdir(Path('.', 'output', model_dir, run))]
    latest_run_dir = max(all_runs, key=os.path.getmtime) if not CHOSEN_RUN_DIR else CHOSEN_RUN_DIR
    flagfile = Path(latest_run_dir, 'flagfile.txt')
    app.run(main, sys.argv + [f'--flagfile={flagfile}'])
