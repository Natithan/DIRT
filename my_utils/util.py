import os
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import distributed as dist

from config import FLAGS, MODEL_RELEVANT_FLAGS
from my_models.wrappers import MLMModelWrapper, MODEL_MAPPING


def cleanup():
    dist.destroy_process_group()


def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def load_pretrained_model_for_SG():
    model_path = FLAGS.saved_pretrained_model_path
    flagfile_path = model_path.replace('best.th', 'flagfile.txt')
    model_FLAGS = deepcopy(FLAGS)
    model_FLAGS(["", f"--flagfile={flagfile_path}"]) # Normally first arg is the name of the file to run, not relevant here
    run_flag_dict = FLAGS.__dict__['__flags'] #TODO refactor this: use getattribute instead of the __dict__ maybe
    model_flag_dict = model_FLAGS.__dict__['__flags']
    for f in MODEL_RELEVANT_FLAGS:
        updated_flags=[]
        if not (run_flag_dict[f].value == model_flag_dict[f].value):
            run_flag_dict[f].value = model_flag_dict[f].value
            updated_flags.append(f)
        if updated_flags:
            print(f"Changed the following flags to that of the pretrained model: {updated_flags}")
    wrapped_model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model],finetune_stage=True)

    # A hack because I renamed one of the models modules :P
    old_state_dict = torch.load(model_path, map_location=torch.device(FLAGS.device_idxs[0]))
    updated_state_dict = OrderedDict(
        (k.replace("model.predictor", "model.lm_head"), v) for k, v in old_state_dict.items())

    wrapped_model.load_state_dict(updated_state_dict)
    unwrapped_model = wrapped_model.model
    return unwrapped_model
