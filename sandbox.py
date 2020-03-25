import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from absl import app
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer
from torch.utils.data import DataLoader

from config import FLAGS

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()

class ToyModel(Model):
    def __init__(self):
        super(ToyModel, self).__init__(vocab=Vocabulary())
        self.net1 = nn.Linear(40000, 10000)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10000, 5)

    def forward(self, x, actual):
        predicted =  self.net2(self.relu(self.net1(x)))
        loss = nn.MSELoss()(predicted,actual)
        return {'loss':loss}



def main(rank, world_size): #TODO figure out why 1 GPU is faster than 2 when using DDP :P
    distributed = world_size > 1
    if distributed:
        setup(rank, world_size)


    device_idx = FLAGS.device_idxs[rank]

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_idx)
    # output_device defaults to device_ids[0]

    optimizer = optim.Adam(model.parameters())
    train_dataset = [{'x': torch.rand(40000), 'actual': torch.rand(5)} for _ in range(12000)]
    loader = DataLoader(train_dataset,
                        batch_size=4,
                        shuffle=True)
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=loader,
                                     num_epochs=10,
                                     distributed=distributed,
                                     world_size=world_size,
                                     cuda_device=device_idx)
    trainer.train()

    if distributed:
        cleanup()


def main_distributed_wrapper(_):
    nb_gpus = len(FLAGS.device_idxs)
    if nb_gpus > 1:
        mp.spawn(main,
                 args=(nb_gpus,),
                 nprocs=nb_gpus,
                 join=True)
    else:
        main(world_size=0,rank=0)
torch.distributed
if __name__ == '__main__':
    app.run(main_distributed_wrapper)