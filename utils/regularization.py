### regularization.py: managing regularization
import torch
from torch import nn
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn import functional as F
import numpy as np

def get_regs_dict():
    return {
        k:v for k,v in globals().items()
        if isinstance(v, type)
            and issubclass(v, RegularizationModule)
            and k[0] != k[0].upper() # exclude classes that start with uppercase
    }

def _verify_dims(shape, dims):
    if dims is None:
        return list(range(len(shape)))
    if shape is None:
        print('Warning: shape is None, cannot verify dims')
        return dims
    out = [i%len(shape) for i in dims]
    assert len(set(out)) == len(out), 'Duplicate dimensions specified'
    out.sort()
    return out
            
class RegularizationModule(nn.Module):
    def __init__(self, coefficient=1, shape=None, dims=None, target=None, keepdims=0, **kwargs):
        super().__init__()

        if isinstance(dims, int):
            dims = [dims]
        
        if isinstance(keepdims, int):
            keepdims = [keepdims]
        
        if shape is None and target is not None:
            shape = target.shape

        self.dims = dims
        self.shape = shape
        self.coefficient = coefficient
        self.target = target
        self.keepdims = _verify_dims(self.shape, keepdims)

    def forward(self, x=None, normalize=False):
        if x is None:
            x = self.target
        if normalize:
            x = x / x.norm()
        return torch.mean(self.function(x) * self.coefficient)

class Compose(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = nn.ModuleList(args)
    def forward(self, x=None):
        return sum([arg(x) for arg in self.args])
    
class Pnorm(RegularizationModule):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = kwargs.get('p', 2)
    def function(self, x):
        return (torch.abs(x)**self.p).mean()
    
class Pnorm_full(RegularizationModule):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = kwargs.get('p', 2)
    def function(self, x):
        # p
        dims = [i for i in range(len(self.shape)) if i not in self.keepdims]
        numel = torch.prod(torch.tensor([self.shape[i] for i in dims]))
        return x.norm(self.p, dim=dims) / numel**(1/self.p)
    
class l1(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = 1
class l2(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = 2
class l4(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = 3
        
class l2_full(Pnorm_full):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = 2
        
class local(RegularizationModule):
    def __init__(self, coefficient=1, shape=None, dims=None, is_global=False, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, **kwargs)
        assert self.shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = _verify_dims(self.shape, dims)
        self.norm = np.mean([self.shape[i] for i in self.dims])#np.prod([self.shape[i] for i in self.dims])**(1/len(self.dims))
        for ind in self.dims:
            i = self.shape[ind]
            v = ((torch.arange(i)-torch.arange(i)[:,None])**2).float()/i**2 # shape = (i,j)
            # v = v/v.max(dim=0)[0]
            self.register_buffer(f'local_pen{ind}', v)
        self.is_global = is_global

    def function(self, x):
        w = x**2

        pen = 0
        for dim in self.dims:
            # get the regularization matrix
            mat = getattr(self, f'local_pen{dim}') # shape = (i,j)
            
            # permute targeted dim to the end
            w_permuted = w.permute(*[i for i in range(len(self.shape)) if i not in [dim]], *[dim])
            
            # reshape while keeping he channel dimension intact
            w_permuted = w_permuted.reshape(self.shape[0], -1, *[self.shape[i] for i in [dim]])

            w_permuted = w_permuted.mean(dim=1)

            ## quadratic form: W^T M W
            temp = w_permuted @ mat
            temp = (temp * w_permuted).sum(dim=1)
            
            pen = pen + temp/(mat.shape[0])
            # # sum over all unpenalized dimensions
            # if self.is_global:
            #     w_permuted = w_permuted.sum(dim=1).unsqueeze(1)
                
            # quadratic form: W^T M W
            # temp = torch.einsum('nci,ij->nj', w_permuted, mat)
            # temp = torch.einsum('ni,nci->n', temp, w_permuted)
            # pen = pen + temp

        return pen/self.norm

class glocal(local):
    def __init__(self, coefficient=1, shape=None, dims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, is_global=True, **kwargs)
        # print('glocal is the same as local')
        
class edge(RegularizationModule):
    def __init__(self, coefficient=1, dims=None, **kwargs):
        super().__init__(coefficient=coefficient, dims=dims, **kwargs)
    def function(self, x):
        self.dims = _verify_dims(x.shape, self.dims)
        w = x**2
        w = w.permute(*self.dims, *[i for i in range(len(self.shape)) if i not in self.dims])
        w = w.reshape(*[self.shape[i] for i in self.dims], -1)
        pen = 0
        for ind in range(len(self.dims)):
            w_permuted_shape = list(range(len(w.shape)))
            w_permuted_shape.remove(ind)
            w_permuted = w.permute(w_permuted_shape+[ind])
            pen = pen + (w_permuted[...,0].mean() + w_permuted[...,-1].mean())/2
        return pen/len(self.dims)

class center(RegularizationModule):
    def __init__(self, coefficient=1, shape=None, dims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, **kwargs)
        assert self.shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = _verify_dims(self.shape, self.dims)
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
    def __init__(self, coefficient=1, kernel=None, dims=None, padding='same', padding_mode='replicate', **kwargs):
        super().__init__(coefficient=coefficient, dims=dims, **kwargs)
        assert kernel is not None, 'Must specify kernel for laplacian'
        if dims is not None:
            assert len(kernel.shape) == len(dims), 'Number of dims must match number of kernel dimensions'
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.conv = [F.conv1d, F.conv2d, F.conv3d][len(kernel.shape)-1]
        self.padding_mode = padding_mode
        self._padding = _calculate_padding(kernel.shape, padding)

    def function(self, x, reduction=torch.mean):
        self.dims = _verify_dims(x.shape, self.dims)
        assert len(self.dims) == len(self.kernel.shape)-2, 'Number of dims must match number of kernel dimensions'
        x = x.permute(*[i for i in range(len(x.shape)) if i not in self.dims], *self.dims)
        x = x.reshape(-1, 1, *[self.shape[i] for i in self.dims]) # shape (N, un-targeted dims, *dims)

        x = F.pad(x, self._padding, mode=self.padding_mode)
        pen = (self.conv(x, self.kernel)**2).mean((0, 1))**0.5 #shape (*dims)
        return reduction(pen)

class localConv(Convolutional):
    def __init__(self, padding='same', padding_mode='constant', **kwargs):
        assert 'target' in kwargs, 'Must specify target for localConv'
        target = kwargs['target']
        shape = target.shape
        dims = _verify_dims(shape, kwargs.get('dims', None))
        # make ndgrid from -1 to 1 of size shape
        grids = torch.meshgrid(
            *[torch.linspace(-1, 1, shape[i]*2) for i in dims]
        )
        # calculate distance from center
        distance = torch.stack(grids).pow(2).sum(dim=0).sqrt()
        
        super().__init__(kernel=distance, padding=padding, padding_mode=padding_mode, **kwargs)
            
    def function(self, x):
        def reduce(x):
            return x.sum()/x.numel()**2
        return super().function(x**2, reduction=reduce)
        
class laplacian(Convolutional):
    #https://en.wikipedia.org/wiki/Discrete_Laplace_operator
    def __init__(self, coefficient=1, dims=None, **kwargs):
        if dims is None:
            if 'shape' in kwargs:
                dims = [i for i in range(len(kwargs['shape']))]
            elif 'target' in kwargs:
                dims = [i for i in range(len(kwargs['target'].shape))]
        if len(dims) == 1:
            kernel = torch.tensor([1.,-2.,1.], dtype=torch.float32)
        elif len(dims) == 2:
            kernel = torch.tensor([[0.25,0.5,0.25],[0.5,-3,0.5],[0.25,0.5,0.25]], dtype=torch.float32)
        elif len(dims) == 3:
            kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                   [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.float32)
        else:
            raise NotImplementedError('Laplacian not implemented for {} dimensions'.format(len(dims)))
        super().__init__(coefficient=coefficient, kernel=kernel, dims=dims, **kwargs)
    
    def function(self, x):
        def reduce(v):
            # norm = v.numel()**((len(v.shape)-1)/len(v.shape))
            norm = np.mean(v.shape)**(len(v.shape)-1)
            return v.sum()/norm
        return super().function(x, reduction=reduce)

def _calculate_padding(kernel_size, padding="same"):
    dilation = [1] * len(kernel_size)
    if isinstance(padding, str):
        _reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
        if padding == 'same':
            for d, k, i in zip(dilation, kernel_size,
                                range(len(kernel_size) - 1, -1, -1)):
                total_padding = d * (k - 1)
                left_pad = total_padding // 2
                _reversed_padding_repeated_twice[2 * i] = left_pad
                _reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)
    else:
        _reversed_padding_repeated_twice = _reverse_repeat_tuple(padding, 2)
    return _reversed_padding_repeated_twice