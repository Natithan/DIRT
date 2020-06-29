import logging as log
import os
import subprocess
import warnings
import torch
from config import FLAGS,process_flags
from absl import app
log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)
from constants import STORAGE_ROOT, SHARED_DATASETS_ROOT
import sys

from jiant.__main__ import main as jiant_main

def main(_):
    process_flags()
    log.info(f'Using GPUs {FLAGS.device_idxs}')
    os.environ["TORCH_HOME"] = STORAGE_ROOT #Load cache files once on arwen, then always read
    os.environ["JIANT_PROJECT_PREFIX"] = f'{STORAGE_ROOT}/output'
    os.environ["JIANT_DATA_DIR"] = f'{SHARED_DATASETS_ROOT}/SuperGLUE'
    os.environ["WORD_EMBS_FILE"] = '-1'
    os.environ["GLOBAL_RO_EXP_DIR"] = f'{STORAGE_ROOT}/output/SG'
    assert len(FLAGS.device_idxs) >0, "No GPU available to run"
    warnings.filterwarnings("ignore", category=UserWarning) #Using newer torch than jiant was built for
    jiant_main(sys.argv[1:])

if __name__ == "__main__":
    app.run(main)
