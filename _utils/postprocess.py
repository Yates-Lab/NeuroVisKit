import torch.nn as nn
import numpy as np
from itertools import cycle

class Loader():
    def __init__(self, ds, cyclic=True, shuffled=False):
        self.ds = ds
        self.inds = np.arange(len(ds))
        if shuffled:
            np.random.shuffle(self.inds)
        self.iter = cycle if cyclic else iter
        self.loader = self.iter(self.inds)
    def __next__(self):
        return self.ds[next(self.loader)]
    def reset(self):
        self.loader = self.iter(self.inds)

def unique(x):
    #maintain order unlike list(set(x))
    out = []
    for i in x:
        if i not in out:
            out.append(i)
    return out

def get_conv_submodules(module, parent_has_weight=False):
    submodules = []
    # Check if the current module has a "weight" property
    has_weight = hasattr(module, 'weight') and (issubclass(type(module), nn.modules.conv._ConvNd) or ("conv" in type(module).__name__.lower()))
    # If the parent or the current module has a "weight" property, don't traverse its children
    if parent_has_weight or has_weight:
        return [module]
    # Otherwise, traverse its children
    for child in module.children():
        submodules.extend(get_conv_submodules(child, has_weight))
    return unique(submodules)

def get_model_conv_weights(model):
    '''
        Get the weights of the convolutional layers of a model.
        Returns a list of lists of weights.
    '''
    weights = []
    for module in model.modules():
        # check if convolutional layer
        if issubclass(type(module), nn.modules.conv._ConvNd):
            weights.append(module.weight.data.cpu().numpy())
    return weights

