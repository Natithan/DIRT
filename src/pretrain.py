# %% Imports
from __future__ import unicode_literals, print_function
import os

import torch
import torch.multiprocessing as mp
from allennlp.common import Tqdm
from allennlp.nn.util import move_to_device

from torch.utils.data import DataLoader, DistributedSampler

from wrappers import MLMModelWrapper, MODEL_MAPPING

from pathlib import Path
import torch.optim as optim

from absl import app
from config import FLAGS, process_flags
import numpy as np

from text_input_pipeline import get_data_dict, get_data_dict_old
# from allennlp.training import Checkpointer

from my_trainer import MyTrainer, MyCheckpointer
from my_utils.util import cleanup, setup
import tqdm

import logging as log
log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa

def get_loader(dataset):
    return DataLoader(dataset,
                        batch_size=FLAGS.d_batch,)


def main(_):
    process_flags()

    if FLAGS.manual_seed:
        set_manual_seeds(FLAGS.manual_seed)

    # Create folders and files to store results and configs
    run_dir = Path(FLAGS.output_folder, FLAGS.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Logging
    log_fh = log.FileHandler(Path(run_dir, 'log.log'))
    log_fmt = log.Formatter("%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p")
    log_fh.setFormatter(log_fmt)
    log.getLogger().addHandler(log_fh)

    #Store the run description, if any
    if FLAGS.description:
        with open(Path(run_dir,'description.txt'),'w') as f:
            f.write(FLAGS.description)
        log.info(f'DESCRIPTION: {FLAGS.description}')
    # Store configuration in same folder as logs and model
    flagfile = Path(run_dir, 'flagfile.txt')
    if os.path.exists(flagfile):
        os.remove(flagfile)
    open(flagfile, "x")
    FLAGS.append_flags_into_file(flagfile)

    if FLAGS.old_pretrain_data:
        data_dict = get_data_dict_old()
    else:
        data_dict = get_data_dict()
    train_dataset, test_dataset, val_dataset = (data_dict[key] for key in
                                                       ('train', 'test', 'val'))
    model = MLMModelWrapper(MODEL_MAPPING[FLAGS.model])
    distributed_wrapper(train,model, run_dir, train_dataset, val_dataset)
    model.cuda(FLAGS.device_idxs[0])

    log.info("Evaluating pretraining performance on test split")
    test_loader = get_loader(test_dataset)
    model.eval()
    batch_generator = iter(test_loader)
    batch_generator = Tqdm.tqdm(
        batch_generator)
    total_metrics = {}
    with torch.no_grad():
        for i, batch in enumerate(batch_generator):
            batch = move_to_device(batch, FLAGS.device_idxs[0])
            if isinstance(batch, torch.Tensor):
                model(batch)
            else:
                model(**batch)
            if i == 0:
                total_metrics = model.get_metrics()
            else:
                total_metrics = {m: total_metrics[m] + model.get_metrics()[m] for m in total_metrics.keys()}
        average_metrics = {k: v/(i+1) for k,v in total_metrics.items()}
        log.info(f"Average test metrics:{average_metrics}")


def set_manual_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def train(rank,world_size,model, run_dir, train_dataset, val_dataset):
    process_flags()
    # If distributed, this is now in one of the threads. Setup makes sure it is in sync with other threads
    distributed = (world_size > 1)
    if distributed:
        setup(rank, world_size)
    cuda_id = FLAGS.device_idxs[rank]
    log.info(f"Using GPU {cuda_id} from GPUs {FLAGS.device_idxs}")
    model.cuda(cuda_id)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    loader = get_loader(train_dataset)
    val_loader = get_loader(val_dataset)
    checkpointer = MyCheckpointer(serialization_dir=run_dir,
                                num_serialized_models_to_keep=FLAGS.num_serialized_models_to_keep)
    trainer = MyTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=loader,
                                     validation_data_loader=val_loader,
                                     patience=FLAGS.patience,
                                     num_epochs=FLAGS.num_epochs,
                                     serialization_dir=run_dir,
                                     model_save_interval=FLAGS.model_save_interval,
                                     checkpointer=checkpointer,
                                     distributed=distributed,
                                     world_size=len(FLAGS.device_idxs),
                                     cuda_device=cuda_id)
    trainer.train()

    if distributed:
        cleanup()


def distributed_wrapper(function,*args):
    print(f'Using GPUs: {FLAGS.device_idxs} unless code changes this flag')
    nb_GPUs = len(FLAGS.device_idxs)
    if nb_GPUs > 1:
        mp.spawn(function,
                 args=(nb_GPUs,) + args,
                 nprocs=nb_GPUs,
                 join=True)
    else:
        function(0,0,*args)

# class TqdmLoggingHandler(log.FileHandler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.tqdm.write(msg)
#             self.flush()
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except:
#             self.handleError(record)
if __name__ == '__main__':
    app.run(main)
