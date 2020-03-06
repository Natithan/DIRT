import os
import numpy as np
from torch import nn


def get_freer_gpu():  # Source: https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_GPUs_free_mem')
    memory_available = [int(x.split()[2]) for x in open('tmp_GPUs_free_mem', 'r').readlines()]
    return int(np.argmax(memory_available))

def get_gpus_with_enough_memory(minimum_memory):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_GPUs_free_mem')
    memory_available = [int(x.split()[2]) for x in open('tmp_GPUs_free_mem', 'r').readlines()]
    used_gpus = np.argwhere(np.array(memory_available) > minimum_memory).squeeze().tolist()
    if not isinstance(used_gpus, Iterable):
        used_gpus = [used_gpus]
    return used_gpus



from collections import OrderedDict, Callable, Iterable


class DefaultOrderedDict(OrderedDict):
    # Source: https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))
