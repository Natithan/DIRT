import logging as log
import os
import subprocess
import warnings

from absl import app
from config import FLAGS
from constants import READ_ONLY_ROOT,WRITE_ROOT
log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa

import sys

from jiant.__main__ import main as jiant_main

def main(_):
    print(f'Using GPUs {FLAGS.device_idxs}')
    os.environ["TORCH_HOME"] = f'/cw/working-arwen/nathan' #Load cache files once on arwen, then always read
    os.environ["JIANT_PROJECT_PREFIX"] = f'{WRITE_ROOT}/output'
    os.environ["JIANT_DATA_DIR"] = f'{READ_ONLY_ROOT}/data'
    os.environ["WORD_EMBS_FILE"] = '-1'
    os.environ["GLOBAL_RO_EXP_DIR"] = f'{READ_ONLY_ROOT}/output/SG'
    assert len(FLAGS.device_idxs) >0, "No GPU available to run"
    warnings.filterwarnings("ignore", category=UserWarning) #Using newer torch than jiant was built for
    jiant_main(sys.argv[1:])

if __name__ == "__main__":
    app.run(main)
