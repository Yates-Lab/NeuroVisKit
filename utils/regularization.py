#%%## regularization.py: managing regularization
import torch
from torch import nn
from NeuroVisKit._utils.utils import is_nn_module
from NeuroVisKit._utils.regularization import _verify_dims
from torch.nn import functional as F
import numpy as np
import warnings

#this makes sure if we move stuff around, users dont need to change imports.
from NeuroVisKit._utils.regularization import *
from NeuroVisKit._utils.regularization import _calculate_padding
def get_regs_dict():
    #returns a dictionary of all available regularization classes
    return {k:v for k,v in globals().items() if inspect.isclass(v) and hasattr(v, '_parent_class') and k[0] != k[0].upper()}

class Regularization(nn.Module):
    """
    Base class for regularization modules
    Used for identifying regularization modules within a model.
    """
    _parent_class = "Regularization"

class Compose(Regularization): #@TODO change to module list or remove entirely
    def __init__(self, *RegModules):
        super().__init__()
        self.args = nn.ModuleList(RegModules)
    def forward(self):
        return sum([arg() for arg in self.args])
    def __getitem__(self, i):
        return self.args[i]
    def append(self, module):
        self.args.append(module)
    def extend(self, modules):
        self.args.extend(modules)
    def __add__(self, modules):
        return self.extend(modules)
    
class ProximalRegularizationModule(Regularization):
    _parent_class = "ProximalRegularizationModule"
    def __init__(self, coefficient=1, lr=1, target=None, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__()
        assert hasattr(target, 'data'), 'Target must be a tensor with a data attribute. Currently, target is of type: '+str(type(target))
        if isinstance(dims, int):
            dims = [dims]
        
        if isinstance(keepdims, int):
            keepdims = [keepdims]
        
        if shape is None and target is not None:
            shape = target.shape
        self.lr = lr
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
    def forward(self):
        assert torch.isfinite(self.target).all(), f'Non-finite values detected in target tensor for regularization type {self.__class__.__name__}. f"percent finite: {torch.isfinite(self.target).float().mean().item()}"'
        with torch.no_grad():
            out = torch.mean(self.proximal())
        if self.log_gradients:
            self.log(out)
        return out
    # def log(self, x):
    #     return gradMagLog.apply(x, self.__class__.__name__)
    
class RegularizationModule(Regularization):
    _parent_class = "RegularizationModule"
    def __init__(self, coefficient=1, shape=None, dims=None, target=None, keepdims=None, process=None, **kwargs):
        super().__init__()
        if is_nn_module(target):
            assert hasattr(target, 'weight')
        else:
            assert target is not None, 'Please set a target object for regularization module.'
        if isinstance(dims, int):
            dims = [dims]
        
        if isinstance(keepdims, int):
            keepdims = [keepdims]
        self._target = target
        
        if shape is None and target is not None:
            shape = self.target.shape

        self.dims = dims
        self.shape = shape
        self.coefficient = coefficient
        self.keepdims = _verify_dims(self.shape, keepdims) if keepdims is not None else []        
        self.log_gradients = kwargs.get('log_gradients', False)
        self.process = process
    def function(self, x):
        raise NotImplementedError
    def forward(self, normalize=False):
        assert torch.isfinite(self.target).all(), f'Non-finite values detected in target tensor for regularization type {self.__class__.__name__}. f"percent finite: {torch.isfinite(self.target).float().mean().item()}"'
        x = self.target
        if self.log_gradients:
            x = self.log(x)
        if normalize:
            x = x / x.norm()
        if self.process is not None:
            x = self.process(x)
        y = torch.mean(self.function(x) * self.coefficient)
        assert y<1e10 and not torch.isnan(y), f'Penalty likely diverged for regularization type {self.__class__.__name__}'
        return y
    @property
    def target(self):
        if is_nn_module(self._target):
            return self._target.weight
        else:
            return self._target
    # def log(self, x):
    #     return gradMagLog.apply(x, self.__class__.__name__)
    
class ActivityRegularization(Regularization):
    _parent_class = "ActivityRegularization"
    def __init__(self, module, coefficient=1, input=False, **kwargs):

        super().__init__()
        self.coefficient = coefficient

        self.module = module
        self.register_buffer('activations', torch.tensor(0.0))

        def get_activation(module, inp, output):
            if input:
                self.activations = inp[0]
            else:
                self.activations = output
            return output
        
        module.register_forward_hook(get_activation)
        
    def forward(self):
        return self.function() * self.coefficient

#%%

class Matrix(RegularizationModule):
    '''
    super class for applying a Matrix penalty to each targeted dimension

    '''
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        assert self.shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = _verify_dims(self.shape, dims)
        self.leftover_dims = [i for i in range(len(self.shape)) if i not in self.dims and i not in self.keepdims]
        self.norm = np.mean([self.shape[i] for i in self.dims])#np.prod([self.shape[i] for i in self.dims])**(1/len(self.dims))

    def function(self, w):
        self.shape = w.shape
        w = w.permute(*self.dims, *self.leftover_dims, *self.keepdims)
        w = w.reshape(
            *[self.shape[i] for i in self.dims],
            -1,
            np.prod([self.shape[i] for i in self.keepdims], dtype=int),
        )
        pen = 0
        for ind, dim in enumerate(self.dims):
            # get the regularization Matrix
            mat = getattr(self, f'pen_mat{dim}') # shape = (i,i)
            # permute targeted dim to the end
            relevant_permute_dims = list(range(len(w.shape)))
            relevant_permute_dims.remove(ind)
            w_permuted = w.permute(*relevant_permute_dims, ind)
            w_permuted = w_permuted.reshape(-1, *w_permuted.shape[-2:])
            temp = torch.einsum('bnd, id, di->n', w_permuted, mat, mat)
            pen = pen + temp
        return pen.sum()
  
class Pnorm(RegularizationModule):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, **kwargs)
        self.p = kwargs['p']
        # self.register_buffer('p', torch.tensor(kwargs.get('p', 2)))
        # self.f = GradMagnitudeLogger("l"+str(self.p))
    def function(self, x):
        # x = self.f(x)
        out = (torch.abs(x)**self.p).sum()

        # out = gradLessDivide.apply(out, x.numel())
        # out = gradLessPower.apply(out, 1/self.p)
        # with torch.no_grad():
        #     out = out/x.numel()
        return out
    
class ProximalPnorm(ProximalRegularizationModule):
    def __init__(self, coefficient=1, target=None, p=1, norm=torch.amax, **kwargs):
        super().__init__(coefficient=coefficient, target=target, **kwargs)
        self.p = p
        self.norm = norm
    def proximal(self):
        # Calculate proximal operator for p-norm
        #grad = norm**(1-p) * abs(x)**(p-1) * torch.sign(x)
        with torch.no_grad():
            scaling = self.norm(torch.abs(self.target), dim=self.dims, keepdim=True) + 1e-8 if self.norm else 1
            x = self.target / scaling
            grads = x.norm(self.p, dim=self.dims, keepdim=True)**(1-self.p) * torch.abs(x)**(self.p-1)
            out = torch.sign(x) * (torch.abs(x) - self.coefficient*self.lr*grads).clamp(min=0)
            self.target.data = out * scaling
        return out.mean([i for i in range(len(out.shape)) if i not in self.keepdims])
    
class Convolutional(RegularizationModule):
    def __init__(self, coefficient=1, kernel=None, dims=None, padding='same', padding_mode='constant', **kwargs):
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

#%%
class activityL1(ActivityRegularization):
    def function(self, *args):
        return self.activations.abs().mean()

class activityL1Sum(ActivityRegularization):
    def function(self, *args):
        return self.activations.abs().sum()
    
class activityL2(ActivityRegularization):
    def function(self, *args):
        return self.activations.pow(2).mean()

class proximalGroupSparsity(ProximalRegularizationModule):
    def __init__(self, coefficient=1, target=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, **kwargs)
        self.p = 2
    def proximal(self):
        with torch.no_grad():
            x = self.target
            norm = x.norm(self.p, dim=self.dims, keepdim=True)
            out = x/norm * (norm - self.coefficient*self.lr).clamp(min=0)
            self.target.data = out
        return out.mean([i for i in range(len(out.shape)) if i not in self.keepdims])

class proximalSparsityDekel(ProximalRegularizationModule):
    def __init__(self, coefficient=1, target=None, groupdim=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, **kwargs)
        self.groupdim = [groupdim] if type(groupdim) is int else groupdim
    def proximal(self):
        #apply shrinkage base on the magnitude of the weights
        with torch.no_grad():
            #max across dims 
            if self.groupdim:
                norm = self.target.norm(2, dim=self.groupdim, keepdim=True)
            else:
                norm = self.target.abs()
            mx = norm.amax((self.dims), keepdim=True)
            shrinkage = (mx / (norm + 1e-6)).clamp(min=0) - 1
            out = torch.sign(self.target) * (torch.abs(self.target) - self.coefficient*self.lr*shrinkage).clamp(min=0)
            self.target.data = out
        return out.mean([i for i in range(len(out.shape)) if i not in self.keepdims])
class proximalL1(ProximalPnorm):
    def __init__(self, coefficient=1e-1, target=None, norm=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, p=1, **kwargs)
        self.norm = norm
    def proximal(self):
        #reimplented for speed in the case of p=1
        with torch.no_grad():
            x = self.target
            norm = self.norm(torch.abs(x), dim=self.dims, keepdim=True) if self.norm is not None else 1
            shrinkage = self.coefficient*self.lr*norm
            out = torch.sign(x) * (torch.abs(x) - shrinkage).clamp(min=0)
            self.target.data = out
        return out.mean([i for i in range(len(out.shape)) if i not in self.keepdims])

class proximalP05(ProximalPnorm):
    def __init__(self, coefficient=1e-1, target=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, p=0.5, **kwargs)
        
class proximalL2(ProximalPnorm):
    def __init__(self, coefficient=1e-2, target=None, **kwargs):
        super().__init__(coefficient=coefficient, target=target, p=2, **kwargs)

class l1(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, p=1, **kwargs)
        
class l2(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, p=2, **kwargs)
        
class l4(Pnorm):
    def __init__(self, coefficient=1, **kwargs):
        super().__init__(coefficient=coefficient, p=4, **kwargs)
        self.p = 4

class max(Matrix):
    '''
    Dan's "max" penalty. Goal: sparsity
    Matrix penalizes each weight element if any other weight is nonzero :
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
    '''
    
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)

        for ind in self.dims:
            i = self.shape[ind]
            v = 1-torch.eye(i) # shape = (i,i)
            self.register_buffer(f'pen_mat{ind}', v)
    
    def function(self, x):
        return super().function(pseudo_huber(x))

class local(Matrix):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        
        for ind in self.dims:
            i = self.shape[ind]
            v = ((torch.arange(i)-torch.arange(i)[:,None])**2).float()/i**2 # shape = (i,j)
            self.register_buffer(f'pen_mat{ind}', v)
        
        # self.f = GradMagnitudeLogger("local")
    def function(self, x):
        return super().function(pseudo_huber(x))

class glocal(local):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        warnings.warn('glocal has been renamed to local. Please use local instead', DeprecationWarning)
        
class fourierLocal(local):
    def __init__(self, coefficient=1e-3, shape=None, dims=None, keepdims=None, **kwargs):
        if shape is None and 'target' in kwargs and kwargs['target'] is not None:
            shape = list(kwargs['target'].shape)
            dims_temp = _verify_dims(shape, dims)
            shape[dims_temp[-1]] = shape[dims_temp[-1]]//2+1
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
    def function(self, x):
        return super().function(torch.abs(torch.fft.fftshift(torch.fft.rfftn(x, dim=self.dims))))
        
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
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):

        if shape is None and 'target' in kwargs and kwargs['target'] is not None:
            shape = list(kwargs['target'].shape)
            
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        # assert self.shape is not None, 'Must specify expected shape of item to be penalized'
        self.dims = _verify_dims(self.shape, self.dims)
        self.leftover_dims = [i for i in range(len(self.shape)) if i not in self.dims and i not in self.keepdims]

        ranges = [torch.linspace(-1, 1, shape[i]) for i in self.dims]
        # center = [shape[i]/2 for i in self.dims]
        grids = torch.meshgrid(*ranges)
        distances = 0
        for g in grids:
            distances = distances + g**2
        # for i, j in zip(grids, center):
        #     distances = distances + (i-j)**2
        distances = distances ** 0.5
        # distances = distances - distances.min()
        self.register_buffer('center_pen', distances)
    def function(self, x):
        w = x**2
        w = w.permute(*self.dims, *self.leftover_dims, *self.keepdims)
        w = w.reshape(
            *[self.shape[i] for i in self.dims],
            -1,
            np.prod([self.shape[i] for i in self.keepdims], dtype=int),
        )

        return (w.mean(-2)*self.center_pen[...,None]).sum()

class fourierCenter(center):
    def __init__(self, coefficient=1, shape=None, dims=None, keepdims=None, **kwargs):
        # if shape is None and 'target' in kwargs and kwargs['target'] is not None:
            # shape = list(kwargs['target'].shape)
            # dims_temp = _verify_dims(shape, dims)
            # shape[dims_temp[-1]] = shape[dims_temp[-1]]//2+1
        super().__init__(coefficient=coefficient, shape=shape, dims=dims, keepdims=keepdims, **kwargs)
        # penalize the DC as well
        self.center_pen = torch.exp(-self.center_pen/5**2) + self.center_pen
    def function(self, x):
        return super().function(torch.abs(torch.fft.fftshift(torch.fft.fftn(x, dim=self.dims))))

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
            return x.mean()
        return super().function(x**2, reduction=reduce)
        
class laplacian(Convolutional):
    #https://en.wikipedia.org/wiki/Discrete_Laplace_operator
    def __init__(self, coefficient=1, dims=None, **kwargs):
        if dims is None:
            if 'shape' in kwargs:
                dims = [i for i in range(len(kwargs['shape']))]
            elif 'target' in kwargs:
                dims = [i for i in range(len(kwargs['target'].shape))]
        elif isinstance(dims, int):
            dims = [dims]
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
# %%