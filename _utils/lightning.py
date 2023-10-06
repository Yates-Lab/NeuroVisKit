import os, sys, shutil, dill
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from NeuroVisKit.utils.process import get_process_dict
from NeuroVisKit._utils.utils import compose

def clean_model_from_wandb(model):
    if hasattr(model, "_wandb_hook_names"):
        del model._wandb_hook_names
    if hasattr(model, "_forward_hooks"):
        for k, v in model._forward_hooks.items():
            s = str(v).lower()
            if "torchhistory" in s or "wandb" in s or "torchgraph" in s:
                del model._forward_hooks[k]
    if hasattr(model, "model") and hasattr(model.model, "_forward_hooks"):
        for k, v in model.model._forward_hooks.items():
            s = str(v).lower()
            if "torchhistory" in s or "wandb" in s or "torchgraph" in s:
                del model.model._forward_hooks[k]
    if hasattr(model, "wrapped_model") and hasattr(model.wrapped_model, "_forward_hooks"):
        for k, v in model.wrapped_model._forward_hooks.items():
            s = str(v).lower()
            if "torchhistory" in s or "wandb" in s or "torchgraph" in s:
                del model.wrapped_model._forward_hooks[k]
    if hasattr(model, "modules"):
        for module in model.modules():
            if hasattr(module, "_forward_hooks"):
                for k, v in module._forward_hooks.items():
                    s = str(v).lower()
                    if "torchhistory" in s or "wandb" in s or "torchgraph" in s:
                        del module._forward_hooks[k]
    return model

def pl_device_format(device):
    """Convert a torch device to the format expected by pytorch lightning.
    **ONLY WORKS FOR SINGLE DEVICE**

    Args:
        device: str or device object describing which device to use

    Returns:
        str: the device formatted for pytorch lightning
    """
    if type(device) == torch.device:
        device = str(device)
    if type(device) == str:
        return ",".join(device.split("cuda:"))[1:] + ','

class PreprocessFunction(nn.Module):
    """Module that applies a list of specified preprocess functions.

    Args:
        preprocess_arr: list of strings specifying the preprocess functions to apply
    """
    def __init__(self, preprocess_arr=[]):
        super().__init__()
        PREPROCESS_DICT = get_process_dict()
        operators = []
        for k, v in PREPROCESS_DICT.items():
            if k in preprocess_arr:
                operators.append(v)
                if isinstance(v, nn.Module):
                    self.add_module(k, v)
        self.operator = compose(*operators)
    def forward(self, x):
        return self.operator(x)

def get_dirnames(config):
    """Returns a dictionary of directory names for the given config.
    Args:
        config (dict): the config dictionary
    Returns:
        dict: a dictionary of directory names
    """
    return {
        'dirname': config["dirname"],
        'checkpoint_dir': os.path.join(config["dirname"], 'models', config['name']),
        'session_dir': os.path.join(config["dirname"], 'sessions', config["session"]),
        'model_path': os.path.join(config["dirname"], 'models', config['name'], 'model.pkl'),
        'config_path': os.path.join(config["dirname"], 'models', config['name'], 'config.json'),
        'ds_dir': os.path.join(config["dirname"], 'sessions', config["session"], 'ds.pkl'),
        'log_dir': os.path.join(config["dirname"], 'models', config['name'], 'lightning_logs'),
    }
    
def prepare_dirs(config):
    """Creates the directories for the given config.

    Args:
        config (dict): the config dictionary
    Returns:
        dirs (dict): a dictionary of directory names
        config (dict): the config dictionary that has been updated with the session metadata.
        session (dict): the session dictionary containing metadata for the intended dataset.
    """
    overwrite = config['overwrite']
    from_checkpoint = config['from_checkpoint']
    print('Device: ', config['device'])
    dirs = get_dirnames(config)
    if os.path.exists(dirs['checkpoint_dir']) and not overwrite and not from_checkpoint:
        print('Directory already exists. Exiting.')
        print('If you want to overwrite, use the -o flag.')
        sys.exit()
    elif overwrite and not from_checkpoint:
        if os.path.exists(dirs['checkpoint_dir']):
            shutil.rmtree(dirs['checkpoint_dir'])
        else:
            print('Directory does not exist (did not overwrite).')
    os.makedirs(dirs['checkpoint_dir'], exist_ok=True)   
    # os.makedirs(dirs['log_dir'], exist_ok=True)
    with open(os.path.join(dirs["session_dir"], 'session.pkl'), 'rb') as f:
        session = dill.load(f)
    config.update({
        'cids': session['cids'],
        'input_dims': session['input_dims'],
        'mu': session['mu'],
    })
    return dirs, config, session