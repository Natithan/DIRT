from torch.utils.data.distributed import DistributedSampler

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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import FLAGS

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):

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
        d_hidden = 5000
        self.net1 = nn.Linear(1000, d_hidden)
        self.relu = nn.ReLU()
        self.net_long = nn.Sequential(*[nn.Linear(d_hidden,d_hidden),nn.ReLU()]*30)
        self.net2 = nn.Linear(d_hidden, 5)

    def forward(self, x, actual):
        print(actual)
        predicted =  self.net2(self.net_long(self.relu(self.net1(x))))
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
    class MyDataset(Dataset):
        def __init__(self,data):
            super().__init__()
            self.data = data
        def __getitem__(self, index):
            return self.data[index]
    train_dataset = MyDataset([{'x': torch.rand(1000), 'actual': (torch.arange(5) + 5*i).to(torch.float)} for i in range(12000)])
    sampler = DistributedSampler(train_dataset)
    loader = DataLoader(train_dataset,
                        batch_size=FLAGS.d_batch,
                        sampler=sampler)
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
if __name__ == '__main__':
    app.run(main_distributed_wrapper)

# DIVISOR

import os

# from allennlp.data.fields import ArrayField
# from allennlp.data.iterators import BasicIterator
#
# print(6)
# import tempfile
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# import torch.multiprocessing as mp
# from absl import app
# print(7)
# from allennlp.data import Vocabulary, Instance
#
# print(8)
# from allennlp.models import Model
# from allennlp.training import Trainer
# from torch.utils.data import DataLoader
# import numpy as np
# from config import FLAGS
#
#
# class ToyModel(Model):
#     def __init__(self):
#         super(ToyModel, self).__init__(vocab=Vocabulary())
#         d_hidden = 5000
#         self.net1 = nn.Linear(1000, d_hidden)
#         self.relu = nn.ReLU()
#         self.net_long = nn.Sequential(*[nn.Linear(d_hidden,d_hidden),nn.ReLU()]*30)
#         self.net2 = nn.Linear(d_hidden, 5)
#
#     def forward(self, x, actual):
#         print(actual)
#         predicted =  self.net2(self.net_long(self.relu(self.net1(x))))
#         loss = nn.MSELoss()(predicted,actual)
#         return {'loss':loss}
#
#
# def main(_):
#
#
#     # create model and move it to device_ids[0]
#     model = ToyModel().to(FLAGS.device_idxs[0])
#     # output_device defaults to device_ids[0]
#
#     optimizer = optim.Adam(model.parameters())
#     train_dataset = [Instance({'x': ArrayField(torch.rand(1000).numpy()), 'actual': ArrayField((torch.arange(5) + 5*i).to(torch.float).numpy())}) for i in range(12000)]
#     iterator = BasicIterator(batch_size=FLAGS.d_batch)
#     trainer = Trainer(model=model,
#                                      optimizer=optimizer,
#                                      iterator=iterator,
#                       train_dataset=train_dataset,
#                                      num_epochs=10,
#                                      cuda_device=FLAGS.device_idxs,
#                                     shuffle=False)
#     trainer.train()
#
#
#
# if __name__ == '__main__':
#     app.run(main)