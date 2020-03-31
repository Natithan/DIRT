import logging as log
import os
import subprocess

from absl import app
from config import FLAGS

log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa

import sys

from jiant.__main__ import main as jiant_main

def main(_):
    print(f'Using GPUs {FLAGS.device_idxs}')
    HOSTNAME = subprocess.check_output('hostname').decode().strip()
    os.environ["JIANT_PROJECT_PREFIX"] = f'/cw/working-{HOSTNAME}/nathan/phd/output'
    jiant_main(sys.argv[1:])

if __name__ == "__main__":
    app.run(main)
