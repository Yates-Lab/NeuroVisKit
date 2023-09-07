### regularization.py: managing regularization
import torch
from torch import nn
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn import functional as F
import numpy as np
from functools import reduce
import inspect
import math

def extract_reg(module, proximal=False):
    def get_reg_helper(module):
        out_modules = []
        for current_module in module.children():
            if issubclass(type(current_module), Regularization) and proximal == issubclass(type(current_module), ProximalRegularization):
                out_modules.append(current_module)
            elif isinstance(current_module, nn.Module):
                    out_modules += get_reg_helper(current_module)
        return out_modules
    return get_reg_helper(module)

def get_regs_dict():
    return {
        k:v for k,v in globals().items()
        if inspect.isclass(v) and issubclass(v, RegularizationModule)
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

class Regularization(nn.Module):
    """
    Base class for regularization modules
    Used for identifying regularization modules within a model.
    """
    def __init__(self):
        super().__init__()

class ProximalRegularization(Regularization):
    def __init__(self, coefficient=1, target=None, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__()
        assert hasattr(target, 'data'), 'Target must be a parameter so we can change it in-place'
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
        self.keepdims = _verify_dims(self.shape, keepdims) if keepdims is not None else []
        
        self.log_gradients = kwargs.get('log_gradients', False)
    def proximal(self):
        """
        Proximal operator -> apply proximal regularization to x
        If easy, should return proximal gradients for visualization purposes
        """
        raise NotImplementedError
    def forward(self, x=None):
        out = torch.mean(self.proximal(x))
        if self.log_gradients:
            self.log(out)
        # print(f'{self.__class__.__name__}: {y}')
        return out

    def log(self, x):
        return gradMagLog.apply(x, self.__class__.__name__)
    
class RegularizationModule(Regularization):
    def __init__(self, coefficient=1, shape=None, dims=None, target=None, keepdims=None, **kwargs):
        super().__init__()
        assert target is not None, 'Please set a target object for regularization module.'
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
        self.keepdims = _verify_dims(self.shape, keepdims) if keepdims is not None else []
        
        self.log_gradients = kwargs.get('log_gradients', False)
    def function(self, x):
        raise NotImplementedError
    def forward(self, x=None, normalize=False):
        if x is None:
            x = self.target
        if self.log_gradients:
            x = self.log(x)
        if normalize:
            x = x / x.norm()
        y = torch.mean(self.function(x) * self.coefficient)
        assert y<1e10 and not torch.isnan(y), f'Penalty likely diverged for regularization type {self.__class__.__name__}'
        # print(f'{self.__class__.__name__}: {y}')
        return y

    def log(self, x):
        return gradMagLog.apply(x, self.__class__.__name__)
        
class Compose(Regularization): #@TODO change to module list or remove entirely
    def __init__(self, *RegModules):
        super().__init__()
        self.args = nn.ModuleList(RegModules)
    def forward(self, x=None):
        return sum([arg(x) for arg in self.args])

class gradMagLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, name, callback=lambda norm,name: print(name+":", norm)):
        ctx.name = name
        ctx.callback = callback
        return input
    @staticmethod
    def backward(ctx, grad_output):
        mag = int(math.log10(torch.abs(grad_output).mean().item()))
        ctx.callback(f"10^{mag}", ctx.name)
        return grad_output, None, None
    
class gradLessOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, callback):
        return callback(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
class gradLessDivide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, divider):
        return input/divider
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
class gradLessMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, multiplier):
        return input*multiplier
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
class gradLessPower(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, power):
        return input**power
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
class GradMagnitudeLogger(nn.Module):
    def __init__(self, name, callback=lambda norm,name: print(name+":", norm)):
        super().__init__()
        self.name = name
        self.callback = callback
    def forward(self, input):
        return gradMagLog.apply(input, self.name, self.callback)
    
class Pnorm(RegularizationModule):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = kwargs.get('p', 2)
        # self.f = GradMagnitudeLogger("l"+str(self.p))
    def function(self, x):
        # x = self.f(x)
        out = (torch.abs(x)**self.p).sum()
        out = gradLessDivide.apply(out, x.numel())
        out = gradLessPower.apply(out, 1/self.p)
        # with torch.no_grad():
        #     out = out/x.numel()
        return out
    
class ProximalPnorm(RegularizationModule):
    def __init__(self, coefficient=1, target=None, p=1, **kwargs):
        super().__init__(coefficient=coefficient, target=target, **kwargs)
        self.p = p
    def proximal(self, x):
        # Calculate proximal operator for p-norm
        norm = x.norm(self.p, dim=self.dims, keepdim=True)
        out = x / (norm + 1e-8) * (norm - self.coefficient).clamp(min=0)
        with torch.no_grad():
            self.target.data = out
        return out.mean([i for i in range(len(out.shape)) if i not in self.keepdims])
    
class ProximalL1(ProximalPnorm):
    def __init__(self, coefficient=1, target=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, p=1, **kwargs)
class ProximalL2(ProximalPnorm):
    def __init__(self, coefficient=1, target=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, p=2, **kwargs)
    
class l1(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, p=1, **kwargs)
class l2(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, p=2, **kwargs)
        
class l1NDNT(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = 1
        # self.f = GradMagnitudeLogger("l1NDNT")
    def function(self, x):
        # x = self.f(x)
        x = torch.abs(x)
        x = x.permute(*self.keepdims, *[i for i in range(len(x.shape)) if i not in self.keepdims])
        x = x.reshape(math.prod(self.keepdims), -1)
        return x.mean(-1).mean()
class l2NDNT(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = 2
        # self.f = GradMagnitudeLogger("l2NDNT")
    def function(self, x):
        # x = self.f(x)
        x = x**self.p
        x = x.permute(*self.keepdims, *[i for i in range(len(x.shape)) if i not in self.keepdims])
        x = x.reshape(math.prod(self.keepdims), -1)
        return x.mean(-1).mean()
    
class l4(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, p=4, **kwargs)
        self.p = 4
        
# class l2_full(Pnorm_full):
#     def __init__(self, coefficient=1, **kwargs):
#         super().__init__(coefficient=coefficient, **kwargs)
#         self.p = 2
        
class local(RegularizationModule):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        assert self.shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = _verify_dims(self.shape, dims)
        self.leftover_dims = [i for i in range(len(self.shape)) if i not in self.dims and i not in self.keepdims]
        self.norm = np.mean([self.shape[i] for i in self.dims])#np.prod([self.shape[i] for i in self.dims])**(1/len(self.dims))
        for ind in self.dims:
            i = self.shape[ind]
            v = ((torch.arange(i)-torch.arange(i)[:,None])**2).float()/i**2 # shape = (i,j)
            self.register_buffer(f'local_pen{ind}', v)
        
        # self.f = GradMagnitudeLogger("local")
    def function(self, x):
        w = x**2
        self.shape = w.shape
        w = w.permute(*self.dims, *self.leftover_dims, *self.keepdims)
        w = w.reshape(
            *[self.shape[i] for i in self.dims],
            -1,
            np.prod([self.shape[i] for i in self.keepdims], dtype=int),
        )
        pen = 0
        for ind, dim in enumerate(self.dims):
            # get the regularization matrix
            mat = getattr(self, f'local_pen{dim}') # shape = (i,j)
            # permute targeted dim to the end
            relevant_permute_dims = list(range(len(w.shape)))
            relevant_permute_dims.remove(ind)
            w_permuted = w.permute(*relevant_permute_dims, ind)
            w_permuted = w_permuted.reshape(-1, *w_permuted.shape[-2:])
            # reshape while keeping he channel dimension intact
            w_permuted = w_permuted.sum(0) #sum over leftover dims
            # w_permuted = gradLessDivide.apply(w_permuted.sum(0), w_permuted.shape[0])
            ## quadratic form: W^T M W
            temp = w_permuted @ mat
            temp = temp * w_permuted
            # temp = temp.sum(1)
            # temp = gradLessDivide.apply(temp.sum(1), temp.shape[1])
            # norm = w.shape[ind]*w.shape[-2]
            pen = pen + temp.sum(1)#gradLessDivide.apply(temp,mat.shape[0])
        return pen.sum()#gradLessDivide.apply(pen.sum(), self.norm)

class glocal(local):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        
class fourierLocal(local):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        if shape is None and 'target' in kwargs and kwargs['target'] is not None:
            shape = list(kwargs['target'].shape)
            dims_temp = _verify_dims(shape, dims)
            shape[dims_temp[-1]] = shape[dims_temp[-1]]//2+1
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
    def function(self, x):
        return super().function(torch.abs(torch.fft.fftshift(torch.fft.rfftn(x, dim=self.dims))))
    
class glocalNDNT(RegularizationModule):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        assert self.shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = _verify_dims(self.shape, dims)
        self.leftover_dims = [i for i in range(len(self.shape)) if i not in self.dims and i not in self.keepdims]
        self.norm = np.mean([self.shape[i] for i in self.dims])#np.prod([self.shape[i] for i in self.dims])**(1/len(self.dims))
        for ind in self.dims:
            i = self.shape[ind]
            v = ((torch.arange(i)-torch.arange(i)[:,None])**2).float()/i**2 # shape = (i,j)
            self.register_buffer(f'local_pen{ind}', v)
        
        # self.f = GradMagnitudeLogger("glocalNDNT")
    def function(self, x):
        # x = self.f(x)
        w = x**2
        w = w.permute(*self.dims, *self.leftover_dims, *self.keepdims)
        w = w.reshape(
            *[self.shape[i] for i in self.dims],
            -1,
            reduce(lambda x,y:x*y, [self.shape[i] for i in self.keepdims], 1),
        ) # reshape to dims, -1, flattened keepdims
        pen = 0
        for ind, dim in enumerate(self.dims):
            mat = getattr(self, f'local_pen{dim}') # shape = (i,j)
            w_permuted_shape = list(range(len(w.shape))) # [*dims.shape, -1, prod(keepdims.shape)]
            w_permuted_shape.remove(ind)
            w_permuted = w.permute(w_permuted_shape+[ind])
            w_permuted = w_permuted.sum(list(range(len(w_permuted.shape)-2)))
            temp = w_permuted @ mat
            temp = torch.einsum('nj,nj->n', temp, w_permuted)
            pen = pen + temp
        return pen.mean()
        
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
        pen = self.conv(x, self.kernel)**2
        pen = pen.sum((0,1))

        # pen = gradLessDivide.apply(pen.sum((0, 1)), np.prod(pen.shape[:2]))
        # pen = gradLessPower.apply(pen, 0.5)
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
            return gradLessDivide.apply(x.sum(), x.numel()**2)
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
            # kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            #                        [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            #                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.float32)
            
            kernel = 1/26*torch.tensor([[[2, 3, 2], [3, 6, 3], [2, 3, 2]],
                                   [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
                                   [[2, 3, 2], [3, 6, 3], [2, 3, 2]]], dtype=torch.float32)
        else:
            raise NotImplementedError('Laplacian not implemented for {} dimensions'.format(len(dims)))
        super().__init__(coefficient=coefficient, kernel=kernel, dims=dims, **kwargs)
    
    def function(self, x):
        def reduce(v):
            # norm = v.numel()**((len(v.shape)-1)/len(v.shape))
            # norm = np.mean(v.shape)**(len(v.shape)-1) #geom mean
            # return gradLessDivide.apply(v.sum(), norm)
            return v.mean()
        return super().function(x, reduction=reduce)

# class huber(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, delta=1e-4):
#         ctx.delta = delta
#         mask = torch.abs(input) < delta
#         ctx.save_for_backward(mask)
#         return torch.where(mask, 0.5*input**2, delta*(torch.abs(input)-0.5*delta))
        
        

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