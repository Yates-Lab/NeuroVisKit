#%%
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import *
import inspect

def get_optim_dict(allCaps=False):
    return {
        k.upper() if allCaps else k:v for k,v in globals().items()
        if inspect.isclass(v)
            and issubclass(v, Optimizer)
            and v is not Optimizer
    }
