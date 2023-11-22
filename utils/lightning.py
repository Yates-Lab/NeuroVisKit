import os, dill, sys, shutil, json, io, contextlib
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
from NeuroVisKit._utils.lightning import PreprocessFunction
from NeuroVisKit._utils.utils import get_module_dict, import_module_by_path, sum_dict_list
import math
import numpy as np
import NeuroVisKit.utils.models as models
from NeuroVisKit.utils.evaluate import EvalModule
from NeuroVisKit.utils.utils import to_device
import warnings
import logging
import importlib.util
# from NeuroVisKit.utils.optimizer import get_optim_dict
# from NeuroVisKit.utils.preprocess import get_process_dict

logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
    
class PLWrapper(pl.LightningModule):
    def __init__(self, wrapped_model=None, lr=1e-3, optimizer=torch.optim.Adam, preprocess_data=PreprocessFunction(), normalize_loss=False, optimizer_kwargs={}):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.model = wrapped_model.model
        self.opt = optimizer
        self.opt_kwargs = optimizer_kwargs
        self.opt_instance = None
        self.learning_rate = lr
        self.preprocess_data = preprocess_data
        assert hasattr(self.wrapped_model, 'cids'), "model must have cids attribute"
        self.cids = self.wrapped_model.cids
        self.register_buffer("per_neuron_loss", torch.zeros((len(self.cids)),))
        self.eval_module = EvalModule(self.cids)
        if normalize_loss:
            raise NotImplementedError("normalize_loss is not implemented yet")
            # self.train_eval_module = TrainEvalModule(self.loss.unit_loss, self.cids)
        self.save_hyperparameters(ignore=['wrapped_model', 'preprocess_data'])
        if hasattr(self.wrapped_model, 'lr'):
            self.wrapped_model.lr = self.learning_rate
        
    def forward(self, x):
        return self.wrapped_model(self.preprocess_data(x))
    
    def configure_optimizers(self):
        self.opt_instance = self.opt(self.wrapped_model.parameters(), lr=self.learning_rate, **self.opt_kwargs)
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
        
    def on_train_start(self):
        if hasattr(self, 'train_eval_module'):
            self.train_eval_module.start(self.trainer.train_dataloader)
        if hasattr(self.model, 'on_train_start'):
            self.model.on_train_start(self)
        self.eval_module.startDS(train_ds=self.trainer.train_dataloader)
        return super().on_train_start()
    
    def on_train_end(self):
        if hasattr(self.model, 'on_train_end'):
            self.model.on_train_end(self)
        return super().on_train_end()
    
    def training_step(self, x, batch_idx=0, dataloader_idx=0):

        if hasattr(self.wrapped_model, 'current_epoch'):
            self.wrapped_model.current_epoch = self.current_epoch
            
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
        self.eval_module(self(x), x)
        self.log("val_loss_poisson", losses["val_loss"], prog_bar=True, on_epoch=True, batch_size=len(x["stim"]))
        if hasattr(self.model, 'on_validation_step'):
            self.model.on_validation_step(self)
        del x
        return losses["val_loss"]
    
    def on_validation_epoch_start(self) -> None:
        if self.opt != torch.optim.LBFGS:
            self.eval_module.reset()
    
    def on_validation_epoch_end(self) -> None:
        if self.opt != torch.optim.LBFGS:
            loss = self.eval_module.closure()
            self.per_neuron_loss = loss.detach()
            self.log("val_loss", -1*torch.mean(loss), prog_bar=True, on_epoch=True)
        if hasattr(self.model, 'on_validation_epoch_end'):
            self.model.on_validation_epoch_end(self)
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