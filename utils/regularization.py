### regularization.py: managing regularization
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

from copy import deepcopy
import numpy as np

def verify_dims(shape, dims):
    if dims is None:
        return list(range(len(shape)))
    out = [i%len(shape) for i in dims]
    assert len(set(out)) == len(out), 'Duplicate dimensions specified'
    out.sort()
    return out
            
class RegularizationModule(nn.Module):
    def __init__(self, shape=None, dims=None, **kwargs):
        super().__init__()
        self.dims = dims
        self.shape = shape
    def forward(self, x):
        return self.function(x)

class l1(RegularizationModule):
    def function(self, x):
        return torch.sum(torch.abs(x)) / x.numel()

class l2(RegularizationModule):
    def function(self, x):
        return torch.sum(x**2) / x.numel()

class local(RegularizationModule):
    def __init__(self, shape=None, dims=None, is_global=False, **kwargs):
        super().__init__(shape=shape, dims=dims, **kwargs)
        assert shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = verify_dims(shape, dims)
        for ind in self.dims:
            i = shape[ind]
            v = ((torch.arange(i)-torch.arange(i)[:,None])**2).float()/i**2 # shape = (i,j)
            self.register_buffer('local_pen'+ind, v)
        self.is_global = is_global
    def function(self, x):
        w = x**2
        w = w.permute(*self.dims, *[i for i in range(len(self.shape)) if i not in self.dims])
        w = w.reshape(*self.dims, -1)
        pen = 0
        for ind, dim in enumerate(self.dims):
            mat = getattr(self, 'local_pen'+dim) # shape = (i,j)
            w_permuted = w.permute(list(range(len(w.shape)))-[ind]+[ind])
            if self.is_global:
                w_permuted = w_permuted.sum(list(range(len(w_permuted.shape)-2))).unsqueeze(0)
            temp = torch.einsum('...ni,ij->nj', w_permuted, mat)
            temp = torch.einsum('ni,...ni->n', temp, w_permuted)
            pen = pen + temp
        return torch.mean(pen)

class glocal(local):
    def __init__(self, shape=None, dims=None, **kwargs):
        super().__init__(shape=shape, dims=dims, is_global=True, **kwargs)
        
class edge(RegularizationModule):
    def __init__(self, dims=None, **kwargs):
        super().__init__(dims=dims, **kwargs)
    def function(self, x):
        self.dims = verify_dims(x.shape, self.dims)
        w = x**2
        w = w.permute(*self.dims, *[i for i in range(len(self.shape)) if i not in self.dims])
        w = w.reshape(*self.dims, -1)
        pen = 0
        for ind in range(len(self.dims)):
            w_permuted = w.permute(list(range(len(w.shape)))-[ind]+[ind])
            pen = pen + (w_permuted[...,0].mean() + w_permuted[...,-1].mean())/2
        return pen/len(self.dims)

class center(RegularizationModule):
    def __init__(self, shape=None, dims=None, **kwargs):
        super().__init__(shape=shape, dims=dims, **kwargs)
        assert shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = verify_dims(shape, self.dims)
        ranges = [torch.linspace(-1, 1, shape[i]) for i in self.dims]
        center = [shape[i]/2 for i in self.dims]
        grids = torch.meshgrid(*ranges)
        distances = 0
        for i, j in zip(grids, center):
            distances = distances + (i-j)**2
        distances = distances ** 0.5
        distances = distances - distances.min()
        self.register_buffer('center_pen', distances)
    def function(self, x):
        w = x**2
        w = w.mean([i for i in range(len(self.shape)) if i not in self.dims])
        return torch.mean(w*self.center_pen)
    
class Convolutional(RegularizationModule):
    def __init__(self, kernel=None, dims=None, **kwargs):
        super().__init__(dims=dims, **kwargs)
        assert kernel is not None, 'Must specify kernel for laplacian'
        assert len(kernel) == len(dims), 'Kernel must have same number of dimensions as dims'
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.conv = [F.conv1d, F.conv2d, F.conv3d][len(dims)-1]
    def function(self, x):
        self.dims = verify_dims(x.shape, self.dims)
        x = x.permute(*[i for i in range(len(x.shape)) if i not in self.dims], *self.dims)
        x = x.reshape(-1, 1, *self.dims) # shape (batch, *dims)
        return (self.conv(x, self.kernel)**2).mean()

class laplacian(Convolutional):
    def __init__(self, dims=None, **kwargs):
        if len(dims) == 1:
            kernel = torch.tensor([[[1,-2,1]]])
        elif len(dims) == 2:
            kernel = torch.tensor([[[0,1,0],[1,-4,1],[0,1,0]]])
        else:
            raise NotImplementedError('Laplacian not implemented for {} dimensions'.format(len(dims)))
        super().__init__(kernel=kernel, dims=dims, **kwargs)