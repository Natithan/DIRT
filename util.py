import os
import numpy as np

def get_freer_gpu(): # Source: https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_GPUs_free_mem')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))