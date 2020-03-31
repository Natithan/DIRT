import os
import torch
from torch import distributed as dist

from config import FLAGS
from wrappers import MLMModelWrapper, MODEL_MAPPING


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

def load_pretrained_model():
    wrapped_model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model]) #TODO get relevant values from loaded model's flagfile
    model_path = FLAGS.saved_pretrained_model_path
    wrapped_model.load_state_dict(torch.load(model_path, map_location=torch.device(FLAGS.device_idxs[0])))

