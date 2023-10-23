import sys, getopt, os, gc
import numpy as np
import random
import contextlib
import torch
import torch.nn as nn
from functools import reduce
import importlib.util
import torch.nn.functional as F
import subprocess as sp
import os
import platform

# def assess_tensors_in_memory():
#     def recursive_tensor_memory(dict_to_recurse):
#         if isinstance(dict_to_recurse, dict):
#             [recursive_tensor_memory(v) for v in dict_to_recurse.values()]
#         elif isinstance(dict_to_recurse, list):
#             [recursive_tensor_memory(v) for v in dict_to_recurse]
#         elif isinstance(dict_to_recurse, torch.Tensor):
#             print(dict_to_recurse.element_size() * dict_to_recurse.nelement() / 1e9)
#     recursive_tensor_memory(globals().copy())
    # for k, v in globals().copy().items():
    #     if issubclass(type(v), torch.Tensor):
    #         print(k, v.element_size() * v.nelement() / 1e9, 'GB', v.device)
def is_nn_module(x):
    return isinstance(x, torch.nn.Module) or isinstance(x, nn.Module)

def auto_device():
    if platform.system().lower() == 'darwin':
        return 'mps'
    memories = get_gpu_memory()
    best = np.argmax(memories)
    if memories[best] > 5000:
        print(f'Using GPU {best} with {memories[best]} MB of free memory')
        return torch.device(f'cuda:{best}')
    else:
        raise ValueError(f'No GPU with more than 5000 MB of free memory found. Free memory: {memories}')
    
def get_device_fancy(device):
    if isinstance(device, str):
        if device.lower() == 'auto':
            return auto_device()
        device = torch.device(device)
    return device

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def sum_dict_list(dlist):
    dsum = {d: [] for d in dlist[0].keys()}
    for d in dlist:
        for k,v in d.items():
            dsum[k].append(v)
    return {k: torch.cat(v, dim=0) for k,v in dsum.items()}

def pad_to_shape(x, shape, padding_mapper=lambda p: [0, p]):
    """Pads a tensor to a given shape.

    Args:
        x is a tensor of any shape
        shape is a tuple of ints of the desired shape
        padding_mapper: a function to map the number of elements to pad to a list of padding values. Defaults to [0, p] which pads p elements to the right.

    Returns:
        _type_: _description_
    """
    assert len(x.shape) == len(shape), "shape must have same number of dimensions as x"
    padding = []
    for x_dim, target_dim in zip(x.shape[::-1], shape[::-1]):
        padding.extend(padding_mapper(max(target_dim - x_dim, 0)))
    return F.pad(x, padding)

def time_embedding(x, num_lags):
    """Embeds a time series into a tensor of shape (time - num_lags, num_lags, n)

    Args:
        x is (time, n)
        num_lags (int): number of lags to embed

    Returns:
        (time - num_lags, num_lags, n)
    """
    tensorlist = [x[i:i+num_lags] for i in range(x.shape[0] - num_lags + 1)]
    assert len(tensorlist) != 0, x.shape
    out = torch.stack(tensorlist, dim=0)
    dims = [0] + list(range(len(out.shape)))[2:] + [1]
    out = out.permute(*dims)
    return out

def get_module_dict(module, all_caps=False, condition=lambda k, v: True):
    """Get a dictionary of all classes in a module that satisfy a condition.

    Args:
        module (module): the module to get the classes from
        all_caps (bool, optional): whether to convert the keys to all caps. Defaults to False.
        condition (function, optional): a function that takes in a key and value and returns whether they should be included. Defaults to trivial condition.

    Returns:
        dict: a dictionary of classes in the module that satisfy the condition
    """
    return {
        (k if not all_caps else k.upper()):v for k,v in module.__dict__.items() if condition(k,v)
    }
    
def compose(*funcs):
    """Compose single argument functions together, where the first function is applied first.
    
    Args:
        *funcs: The functions to compose.

    Returns:
        The composed function.
    """
    return lambda x: reduce(lambda acc, f: f(acc), funcs[::-1], x)

@contextlib.contextmanager
def print_off():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def joinCWD(*paths):
    return os.path.join(os.getcwd(), *paths)

def isInteractive():
    """Check if the current session is interactive or not."""
    #can use __name__  != "__main__" or hasattr(__main__, 'get_ipython')
    return bool(getattr(sys, 'ps1', sys.flags.interactive))

def get_opt_dict(opt_config, default=None):
    """
        Get the requested options from command line as a dictionary
        opt_config: takes in a list of tuples, where each tuple is one of four options:
        A. options that take an argument from the command line
            1. (shorthand:, fullname=, func) applies a function on the argument
            2. (shorthand:, fullname=) gives the argument as a string
        B. options that do not take an argument from the command line
            3. (shorthand, fullname, value) sets argument to value if the option is present
            4. (shorthand, fullname) sets argument to True if the option is present
    """
    argv = sys.argv[1:]
    opt_names = [i[1] for i in opt_config if i[1]]
    opt_shorthand = [i[0] for i in opt_config if i[0]]
    opts, args = getopt.getopt(argv, "".join(opt_shorthand), opt_names)
    opts_dict = {}
    opt_names = ['--'+str(i[1]).replace('=', '') for i in opt_config]
    opt_shorthand = ['-'+str(i[0]).replace(':', '') for i in opt_config]
    for opt, arg in opts:
        opt_tuple = opt_config[opt_names.index(opt) if opt in opt_names else opt_shorthand.index(opt)]
        opt_name = (opt_tuple[1] or opt_tuple[0])
        if '=' in opt_name or ':' in opt_name:
            opt_func = opt_tuple[2] if len(opt_tuple) > 2 else lambda x: x
            opts_dict[opt_name.replace('=', '').replace(':', '')] = (opt_func or (lambda x: x))(arg)
        else:
            opts_dict[opt_name] = opt_tuple[2] if len(opt_tuple) > 2 else True
    if default is not None:
        default.update(opts_dict)
        return default
    return opts_dict

def seed_everything(seed, noprint=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.cuda.manual_seed(str(seed))
    if not noprint:
        print(f'Seeding with {seed}')

def memory_clear():
    '''
        Clear unneeded memory.
    '''
    torch.cuda.empty_cache()
    gc.collect()
    
def import_module_by_path(module_path, module_name="custom_models"):
    """Import a module by its path.

    Args:
        module_path (str): the path to the module
        module_name (str, optional): the name of the module. Defaults to "custom_models".

    Returns:
        module: the imported module
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    return custom_module