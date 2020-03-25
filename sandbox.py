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


def setup():

    # initialize the process group
    dist.init_process_group("nccl", init_method='env://')

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()

class ToyModel(Model):
    def __init__(self):
        super(ToyModel, self).__init__(vocab=Vocabulary())
        d_hidden = 3000
        self.net1 = nn.Linear(1000, d_hidden)
        self.relu = nn.ReLU()
        self.net_long = nn.Sequential(*[nn.Linear(d_hidden,d_hidden),nn.ReLU()]*30)
        self.net2 = nn.Linear(d_hidden, 5)

    def forward(self, x, actual):
        predicted =  self.net2(self.net_long(self.relu(self.net1(x))))
        loss = nn.MSELoss()(predicted,actual)
        return {'loss':loss}



def main(_): #TODO figure out why 1 GPU is faster than 2 when using DDP :P
    #TODO make sure no zombie process when using torch distributed launch
    gpus = FLAGS.device_idxs
    world_size=len(gpus)
    distributed = world_size > 1
    setup()
    device_idx = gpus[FLAGS.local_rank]

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_idx)
    # output_device defaults to device_ids[0]

    optimizer = optim.Adam(model.parameters())
    train_dataset = [{'x': torch.rand(1000), 'actual': torch.rand(5)} for _ in range(12000)]
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


if __name__ == '__main__':
    app.run(main)