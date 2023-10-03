import os, dill, sys, shutil, json, io, contextlib
import re
from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler, Dataset
import lightning as pl
from lightning.pytorch.callbacks import LearningRateFinder
import tqdm
from zmq import has
from NeuroVisKit._utils.lightning import PreprocessFunction
from NeuroVisKit._utils.utils import get_module_dict, import_module_by_path, sum_dict_list
import math
import numpy as np
import NeuroVisKit.utils.models as models
from NeuroVisKit.utils.utils import to_device
import warnings
import logging
import importlib.util
from NeuroVisKit.utils.optimizer import get_optim_dict
from NeuroVisKit.utils.process import get_process_dict

logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class EvalModule:
    """Module that evaluates the log-likelihood of a model on a given dataset.
    This is used for trainers that have a built in validation loop. Otherwise you can use an end-to-end alternative that loops through the data.
    """
    def __init__(self, unit_loss, cids):
        self.loss = unit_loss
        self.cids = cids
        self.tsum, self.rsum, self.llsum = 0, 0, 0
    def reset(self):
        self.tsum, self.rsum, self.llsum = 0, 0, 0
    def step(self, x, rpred):
        self.llsum += self.loss(
            rpred,
            x["robs"][:, self.cids],
            data_filters=x["dfs"][:, self.cids],
            temporal_normalize=False
        )
        self.tsum += x["dfs"][:, self.cids].sum(dim=0)
        self.rsum += (x["dfs"][:, self.cids]*x["robs"][:, self.cids]).sum(dim=0)
    def closure(self):
        assert type(self.llsum) is not int, "EvalModule has not been called yet"
        assert type(self.tsum) is not int, "EvalModule has not been called yet"
        assert type(self.rsum) is not int, "EvalModule has not been called yet"
        zeros = torch.where((self.rsum == 0))[0].tolist() #check for zeros
        if zeros:
            print(f"(ignore if just sanity checking) no spikes detected for neurons {zeros}. Check your cids, data and datafilters.")
            self.reset()
            return torch.tensor(0)
        LLneuron = self.llsum/self.rsum.clamp(1)
        rbar = self.rsum/self.tsum.clamp(1)
        LLnulls = torch.log(rbar)-1
        LLneuron = -LLneuron - LLnulls
        LLneuron/=np.log(2)
        return LLneuron

class TrainEvalModule(nn.Module):
    """Module that evaluates the log-likelihood of a model on a given dataset.
    This is used for training.
    """
    def __init__(self, unit_loss, cids):
        super().__init__()
        self.loss = unit_loss
        self.cids = cids
    def start(self, train_dataloader):
        self.train_dataloader = train_dataloader
        sum_spikes, count = 0, 0
        for b in tqdm.tqdm(train_dataloader, desc="Preparing normalization for loss."):
            count = count + b["dfs"][:, self.cids].sum(dim=0)
            sum_spikes = sum_spikes + (b["dfs"][:, self.cids]*b["robs"][:, self.cids]).sum(dim=0)
        self.register_buffer("mean_spikes", sum_spikes/count.clamp(1))
    def __call__(self, rpred, batch):
        llsum = self.loss(
            rpred,
            batch["robs"][:, self.cids],
            data_filters=batch["dfs"][:, self.cids],
            temporal_normalize=False
        )
        spike_sum = (batch["dfs"][:, self.cids]*batch["robs"][:, self.cids]).sum(dim=0)
        LLneuron = llsum/spike_sum.clamp(1)
        LLnulls = torch.log(self.mean_spikes)-1
        device = spike_sum.device
        LLneuron = -LLneuron.to(device) - LLnulls.to(device)
        LLneuron/=np.log(2)
        return -LLneuron
    
class PLWrapper(pl.LightningModule):
    def __init__(self, wrapped_model=None, lr=1e-3, optimizer=torch.optim.Adam, preprocess_data=PreprocessFunction(), normalize_loss=True):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.model = wrapped_model.model
        self.opt = optimizer
        self.opt_instance = None
        self.learning_rate = lr
        self.preprocess_data = preprocess_data
        assert hasattr(self.wrapped_model, 'cids'), "model must have cids attribute"
        self.cids = self.wrapped_model.cids
        self.eval_module = EvalModule(self.loss.unit_loss, self.cids)
        if normalize_loss:
            self.train_eval_module = TrainEvalModule(self.loss.unit_loss, self.cids)
        self.save_hyperparameters(ignore=['wrapped_model', 'preprocess_data'])
        if hasattr(self.wrapped_model, 'lr'):
            self.wrapped_model.lr = self.learning_rate
        
    def forward(self, x):
        return self.wrapped_model(self.preprocess_data(x))
    
    def configure_optimizers(self):
        self.opt_instance = self.opt(self.wrapped_model.parameters(), lr=self.learning_rate)
        return self.opt_instance
    
    def update_lr(self, lr=None):
        if lr is not None:
            self.learning_rate = lr
            if hasattr(self.wrapped_model, 'lr'):
                self.wrapped_model.lr = lr
            print(f"Updating learning rate to {self.learning_rate:.5f}")
            for g in self.opt_instance.param_groups:
                g['lr'] = self.learning_rate
        else:
            raise ValueError("lr must be specified, yet it is None")
    
    def on_train_epoch_start(self):
        if hasattr(self, 'train_eval_module'):
            self.train_eval_module.start(self.trainer.train_dataloader)
        return super().on_train_epoch_start()
    
    def training_step(self, x, batch_idx=0, dataloader_idx=0):
        x = self.preprocess_data(x)
        if hasattr(self, 'train_eval_module'):
            losses = self.wrapped_model.training_step(x, alternative_loss_fn=self.train_eval_module)
        else:
            losses = self.wrapped_model.training_step(x)
        self.log("train_loss", losses['train_loss'], prog_bar=True, on_epoch=True, batch_size=len(x["stim"]), on_step=True)
        if "reg_loss" in losses.keys():
            self.log("reg_loss", losses['reg_loss'], prog_bar=True, on_step=True, batch_size=len(x["stim"]))
        del x
        return losses['loss']
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        out = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if hasattr(self.wrapped_model, 'compute_proximal_reg_loss'):
            preg_loss = self.wrapped_model.compute_proximal_reg_loss()
            self.log("proximal_reg_loss", float(preg_loss), prog_bar=True, on_step=True)
        return out
    
    def validation_step(self, x, batch_idx=0, dataloader_idx=0):
        x = self.preprocess_data(x)
        losses = self.wrapped_model.validation_step(x)
        self.eval_module.step(x, self(x))
        self.log("val_loss_poisson", losses["val_loss"], prog_bar=True, on_epoch=True, batch_size=len(x["stim"]))
        del x
        return losses["val_loss"]
    
    def on_validation_epoch_start(self) -> None:
        if self.opt != torch.optim.LBFGS:
            self.eval_module.reset()
    
    def on_validation_epoch_end(self) -> None:
        if self.opt != torch.optim.LBFGS:
            self.log("val_loss", -1*self.eval_module.closure().mean(), prog_bar=True, on_epoch=True)
        with torch.no_grad():
            self._logging()
    
    def _logging(self):
        if hasattr(self.model, 'logging'):
            self.model.logging(self)
        
    @property
    def loss(self):
        return self.wrapped_model.loss
    @loss.setter
    def loss(self, loss):
        self.wrapped_model.loss = loss
    
    @staticmethod
    def load_from_config_path(config_path, ignore_model_capitalization=False):
        with open(config_path, 'rb') as f:
            config = json.load(f)
        if config["checkpoint_path"][-3:] == "pkl":
            with open(config["checkpoint_path"], 'rb') as f:
                return dill.load(f)
        with open(config["checkpoint_path"], 'rb') as f:
            cp = torch.load(f)
        if "custom_models_path" in config and config["custom_models_path"] is not None:
            module = import_module_by_path(config["custom_models_path"], "custom_models")
        else:
            module = models
        modelname = config["model"] if not ignore_model_capitalization else config["model"].upper()
        model = get_module_dict(module, all_caps=ignore_model_capitalization, condition=lambda k, v: hasattr(v, "fromConfig"))[modelname].fromConfig(config)
        plmodel = PLWrapper(wrapped_model=model, preprocess_data=PreprocessFunction(config["dynamic_preprocess"]), **cp["hyper_parameters"]) 
        plmodel.load_state_dict(cp["state_dict"])
        return plmodel
    
def get_model(config):
    lr = config["lr"] if "lr" in config.keys() else 3e-4
    lr = lr if config["trainer"] != "lbfgs" else 1
    optimizer = get_optim_dict(allCaps=config['ignore_capitalization'])[config["trainer"] if not config['ignore_capitalization'] else config["trainer"].upper()]
    preprocess = PreprocessFunction(config["dynamic_preprocess"])
    if "custom_models_path" in config and config["custom_models_path"] is not None:
        module = import_module_by_path(config["custom_models_path"], "custom_models")
    else:
        module = models
    ignore_model_capitalization = config["ignore_model_capitalization"] if "ignore_model_capitalization" in config else False
    modelname = config["model"] if not ignore_model_capitalization else config["model"].upper()
    model = get_module_dict(module, all_caps=ignore_model_capitalization, condition=lambda k, v: hasattr(v, "fromConfig"))[modelname].fromConfig(config)
    return PLWrapper(model, lr=lr, optimizer=optimizer, preprocess_data=preprocess)

def get_fix_dataloader(ds, inds, batch_size=1, device=None):
    sampler = BatchSampler(
        SubsetRandomSampler(inds),
        batch_size=batch_size,
        drop_last=True
    )
    if device is None or device == "cpu":
        num_workers=os.cpu_count()//2
        dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=num_workers)#, pin_memory=device is not None, pin_memory_device=device)
    else:
        dl = DataLoader(ds, sampler=sampler, batch_size=None)
    return dl

def prepare_pretrained_core(model, config):
    if config['pretrained_core'] is not None:
            with open(os.path.join(config["dirname"], 'models', config['pretrained_core'], 'model.pkl'), 'rb') as f:
                model_pretrained = dill.load(f).model.core
                for i in model_pretrained.parameters():
                    i.requires_grad = False
                model.model.core = model_pretrained
                
def defrost_core(model, config):
    if config['defrost']:
        for i in model.model.core.parameters():
            i.requires_grad = True
    
def prepare_model(config, dirs):
    if config["from_checkpoint"]:
        try:
            model = PLWrapper.load_from_config_path(dirs["config_path"])
        except Exception as e:
            print("Could not load from PyTorch checkpoint. Using pickled version.")
            try:
                with open(dirs["model_path"], 'rb') as f:
                    model = dill.load(f)
            except Exception as e:
                print("Could not load from pickled model. Exiting.")
                raise e
        prepare_pretrained_core(model, config)
    else:
        model = get_model(config)
    defrost_core(model, config)
    return model

# class PrintOff():
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout
    
# class LRFinder(LearningRateFinder):
#     def __init__(self, early_stop_threshold=None, num_training_steps=100, mode="exponential", **kwargs):
#         super().__init__(early_stop_threshold=early_stop_threshold, num_training_steps=num_training_steps, mode=mode, **kwargs)
#         self.best_val_loss = torch.inf
#     def on_train_epoch_start(self, trainer, pl_module):
#         if trainer.current_epoch == 0:
#             pl_module.update_lr(self.optimal_lr.suggestion())
#         if  "val_loss" in trainer.logged_metrics and trainer.logged_metrics["val_loss"] >= self.best_val_loss:
#             with print_off():
#                 self.lr_find(trainer, pl_module)
#             if hasattr(pl_module, 'update_lr'):
#                 pl_module.update_lr(self.optimal_lr.suggestion())
#             else:
#                 print('model does not have update_lr method. Please write one or ensure LR is changing... ')
#         else:
#             if "val_loss" in trainer.logged_metrics:
#                 self.best_val_loss = trainer.logged_metrics["val_loss"]

# def get_dirnames(config):
#     return {
#         'dirname': config["dirname"],
#         'checkpoint_dir': os.path.join(config["dirname"], 'models', config['name']),
#         'session_dir': os.path.join(config["dirname"], 'sessions', config["session"]),
#         'model_path': os.path.join(config["dirname"], 'models', config['name'], 'model.pkl'),
#         'config_path': os.path.join(config["dirname"], 'models', config['name'], 'config.json'),
#         'ds_dir': os.path.join(config["dirname"], 'sessions', config["session"], 'ds.pkl'),
#         'log_dir': os.path.join(config["dirname"], 'models', config['name'], 'lightning_logs'),
#     }
    
# def prepare_dirs(config):
#     overwrite = config['overwrite']
#     from_checkpoint = config['from_checkpoint']
#     print('Device: ', config['device'])
#     dirs = get_dirnames(config)
#     if os.path.exists(dirs['checkpoint_dir']) and not overwrite and not from_checkpoint:
#         print('Directory already exists. Exiting.')
#         print('If you want to overwrite, use the -o flag.')
#         sys.exit()
#     elif overwrite and not from_checkpoint:
#         if os.path.exists(dirs['checkpoint_dir']):
#             shutil.rmtree(dirs['checkpoint_dir'])
#         else:
#             print('Directory does not exist (did not overwrite).')
#     os.makedirs(dirs['checkpoint_dir'], exist_ok=True)   
#     # os.makedirs(dirs['log_dir'], exist_ok=True)
#     with open(os.path.join(dirs["session_dir"], 'session.pkl'), 'rb') as f:
#         session = dill.load(f)
#     config.update({
#         'cids': session['cids'],
#         'input_dims': session['input_dims'],
#         'mu': session['mu'],
#     })
#     return dirs, config, session
