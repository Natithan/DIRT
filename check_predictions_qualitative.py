import os
import random

from models.wrappers import MLMModelWrapper, MODEL_MAPPING

import sys

import torch
from absl import app

from config import FLAGS
from pathlib import Path

from text_input_pipeline import GutenbergReader
from util import get_gpus_with_enough_memory


def main(_):
    FLAGS.device_idxs = get_gpus_with_enough_memory(8000) #Hack to not use flagvalue of og model when pertaining to GPU usage
    reader = GutenbergReader()
    data_dict = reader.get_data_dict()
    train_dataset, test_dataset, val_dataset, vocab = (data_dict[key] for key in
                                                       ('train', 'test', 'val', 'vocab'))
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model], vocab)
    model = model.cuda(f'cuda:{FLAGS.device_idxs[0]}') #TODO fix RuntimeError: cublas runtime error : resource allocation failed at /pytorch/aten/src/THC/THCGeneral.cpp:216 THCudaCheck FAIL file=/pytorch/aten/src/THC/THCCachingHostAllocator.cpp line=278 error=700 : an illegal memory access was encountered

    best_val_run_path = Path(latest_run_dir, 'best.th')
    model_states = [Path(latest_run_dir, m) for m in os.listdir(latest_run_dir) if 'model_state_epoch_' in m]
    latest_run_path = max(model_states, key=os.path.getmtime)

    for trained_model_path in (best_val_run_path, latest_run_path):
        print(f'Testing {trained_model_path}')
        model.load_state_dict(torch.load(trained_model_path))
        model = model.cuda(f'cuda:{FLAGS.device_idxs[0]}')
        model.eval()  # Set to eval mode
        for name, dataset in zip(('Train', 'Test', 'Val'), (train_dataset, test_dataset, val_dataset)):
            print(f'Testing {name}')
            instances = random.sample(dataset, 3)
            prediction = model.forward_on_instances(
                instances)
            input_texts = [tokens_to_mask_aware_text(model.token_indexer,
                                                     model.token_indexer.convert_ids_to_tokens(
                                                         instance.fields['input_ids'].array),
                                                     prediction[batch_sample]['mask'])
                           for batch_sample, instance in enumerate(instances)]
            predicted_texts = [tokens_to_mask_aware_text(model.token_indexer,
                                                         model.token_indexer.decode(
                                                             prediction[batch_sample]['vocab_scores'].argmax(1)),
                                                         prediction[batch_sample]['mask'])
                               for batch_sample in range(len(prediction))]
            prediction = model.forward_on_instances(
                instances)
            for input_text, predicted_text in zip(input_texts, predicted_texts):
                print(f'Input: {input_text}')
                print(f'Prediction: {predicted_text}')
                print(' ')


# Copied from HF transformers
def tokens_to_mask_aware_text(tokenizer, filtered_tokens, mask,
                              skip_special_tokens=False, clean_up_tokenization_spaces=True):
    # To avoid mixing byte-level and unicode for byte-level BPT
    # we need to build string separatly for added tokens and byte-level tokens
    # cf. https://github.com/huggingface/transformers/issues/1133
    sub_texts = []
    current_sub_text = []
    for token, isMasked in zip(filtered_tokens,mask):
        if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
        if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
                sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text)) #TODO figure out here how to add mask_indication without messing up decoding
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
    latest_run_dir = max(all_runs, key=os.path.getmtime)
    flagfile = Path(latest_run_dir, 'flagfile.txt')
    app.run(main, sys.argv + [f'--flagfile={flagfile}'])
