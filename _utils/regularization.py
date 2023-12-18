import torch
import torch.nn as nn
import inspect
import math
from torch.nn.modules.utils import _reverse_repeat_tuple

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
    

def get_regs_dict():
    #returns a dictionary of all available regularization classes
    return {
        k:v for k,v in globals().items()
        if inspect.isclass(v) and hasattr(v, '_parent_class') and k[0] != k[0].upper() 
    }
    # only include classes that are subclasses of Regularization
    #NOTE the ugly parent class attribute is a solution for import class detection issues with python
    # exclude classes that start with uppercase because they are abstract classes

def pseudo_huber(x, delta=1):
    return delta**2 * (torch.sqrt(1 + (x/delta)**2) - 1)

def _verify_dims(shape, dims):
    if dims is None:
        return list(range(len(shape)))
    elif isinstance(dims, int):
        dims = [dims]
    if shape is None:
        print('Warning: shape is None, cannot verify dims')
        return dims
    out = [i%len(shape) for i in dims]
    assert len(set(out)) == len(out), 'Duplicate dimensions specified'
    out.sort()
    return out

def extract_reg(module, proximal=False):
    def get_reg_helper(module):
        out_modules = []
        for current_module in module.children():
            if hasattr(current_module, '_parent_class'): # critical for extract_reg to work when importing from different paths
                isproximal = current_module._parent_class.__name__ == 'ProximalRegularizationModule'
                isregmodule = current_module._parent_class.__name__ == 'RegularizationModule'
                isactivity = current_module._parent_class.__name__ == 'ActivityRegularization'
                if isproximal or isregmodule:
                    if proximal == isproximal and current_module.coefficient:
                        out_modules.append(current_module)
                if isactivity and current_module.coefficient:
                    out_modules.append(current_module)
            elif issubclass(type(current_module), nn.Module):
                out_modules += get_reg_helper(current_module)
        return out_modules
    return get_reg_helper(module)