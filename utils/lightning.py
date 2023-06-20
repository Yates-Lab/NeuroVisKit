import os, dill, sys, shutil, json, io, contextlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler, Dataset
import lightning as pl
from lightning.pytorch.callbacks import LearningRateFinder
import math
import numpy as np
import utils.get_models as get_models
import moten

def time_embedding(x, num_lags):
    # x is (time, n)
    # output is (time - num_lags, num_lags, n)
    tensorlist = [x[i:i+num_lags] for i in range(x.shape[0] - num_lags + 1)]
    assert len(tensorlist) != 0, x.shape
    out = torch.stack(tensorlist, dim=0)
    return out.permute(0,2,1).reshape(len(out), -1)

def get_embed_function(num_lags=36):
    def embed(x):
        x["stim"] = time_embedding(x["stim"].flatten(1), num_lags)
        x["dfs"] = x["dfs"][num_lags-1:]
        x["robs"] = x["robs"][num_lags-1:]
        return x
    return embed

def binarize(x):
    x["robs"][x["robs"] > 0] = 1
    return x

class GaborPreprocess(nn.Module):
    def __init__(self, hw, fps=240):
        super().__init__()
        self.hw = hw
        self.pyramid = moten.get_default_pyramid(vhsize=hw, fps=fps)
    def forward(self, x):
        bsize = x["stim"].shape[0]
        with print_off():
            x["stim"] = self.pyramid.project_stimulus(x["stim"].reshape(bsize, *self.hw).cpu().numpy())
        x["stim"] = torch.from_numpy(x["stim"].reshape(bsize, -1)).to(x["robs"].device)
        return x
class Trim(nn.Module):
    def forward(self, x):
        if "trimmed" in x:
            return x
        x["robs"] = x["robs"][35:]
        x["dfs"] = x["dfs"][35:]
        x["trimmed"] = True
        return x
    
class PreprocessFunction(nn.Module):
    def __init__(self, preprocess_arr=[]):
        super().__init__()
        self.PREPROCESS_DICT = {
            "binarize": binarize,
            "gabor": GaborPreprocess((70, 70), fps=240),
            "trim": Trim(),
            "time_embed": get_embed_function(36),
        }
        for k, v in self.PREPROCESS_DICT.items():
            if isinstance(v, nn.Module):
                self.add_module(k, v)
        self.operators = [k for k in self.PREPROCESS_DICT.keys() if k in preprocess_arr]
    def forward(self, x):
        for op in self.operators:
            x = self.PREPROCESS_DICT[op](x)
        return x

class EvalModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tsum, self.rsum, self.llsum = 0, 0, 0
    def reset(self):
        self.tsum, self.rsum, self.llsum = 0, 0, 0
    def step(self, x):
        rpred = self.model(x)
        self.llsum += self.model.loss.unit_loss(
            rpred,
            x["robs"][:, self.model.cids],
            data_filters=x["dfs"][:, self.model.cids],
            temporal_normalize=False
        )
        self.tsum += x["dfs"][:, self.model.cids].sum(dim=0)
        self.rsum += (x["dfs"][:, self.model.cids]*x["robs"][:, self.model.cids]).sum(dim=0)
    def closure(self):
        assert type(self.llsum) is not int, "EvalModule has not been called yet"
        assert type(self.tsum) is not int, "EvalModule has not been called yet"
        assert type(self.rsum) is not int, "EvalModule has not been called yet"
        LLneuron = self.llsum/self.rsum.clamp(1)
        rbar = self.rsum/self.tsum.clamp(1)
        LLnulls = torch.log(rbar)-1
        LLneuron = -LLneuron - LLnulls
        LLneuron/=np.log(2)
        return LLneuron

class PLWrapper(pl.LightningModule):
    def __init__(self, wrapped_model=None, lr=1e-3, optimizer=torch.optim.Adam, preprocess_data=PreprocessFunction()):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.model = wrapped_model.model
        self.opt = optimizer
        self.opt_instance = None
        self.learning_rate = lr
        self.preprocess_data = preprocess_data
        self.eval_module = EvalModule(wrapped_model)
        if hasattr(self.wrapped_model, 'cids'):
            self.cids = self.wrapped_model.cids
        self.save_hyperparameters(ignore=['wrapped_model', 'preprocess_data'])
        
    def forward(self, x):
        return self.wrapped_model(self.preprocess_data(x))
    
    def configure_optimizers(self):
        self.opt_instance = self.opt(self.wrapped_model.parameters(), lr=self.learning_rate)
        return self.opt_instance
    
    def update_lr(self, lr=None):
        if lr is not None:
            self.learning_rate = lr
        print(f"Updating learning rate to {self.learning_rate:.5f}")
        for g in self.opt_instance.param_groups:
            g['lr'] = self.learning_rate
    
    def training_step(self, x, batch_idx=0, dataloader_idx=0):
        x = self.preprocess_data(x)
        losses = self.wrapped_model.training_step(x)
        self.log("train_loss", losses['train_loss'], prog_bar=True, on_epoch=True, batch_size=len(x["stim"]))
        if "reg_loss" in losses.keys():
            self.log("reg_loss", losses['reg_loss'], prog_bar=True, on_step=True, batch_size=len(x["stim"]))
        del x
        return losses['loss']
    
    def validation_step(self, x, batch_idx=0, dataloader_idx=0):
        x = self.preprocess_data(x)
        losses = self.wrapped_model.validation_step(x)
        self.eval_module.step(x)
        self.log("val_loss_poisson", losses["val_loss"], prog_bar=True, on_epoch=True, batch_size=len(x["stim"]))
        del x
        return losses["val_loss"]
    
    def on_validation_epoch_start(self) -> None:
        if self.opt != torch.optim.LBFGS:
            self.eval_module.reset()
    
    def on_validation_epoch_end(self) -> None:
        if self.opt != torch.optim.LBFGS:
            self.log("val_loss", -1*self.eval_module.closure().mean(), prog_bar=True, on_epoch=True)
        
    @property
    def loss(self):
        return self.wrapped_model.loss
    @loss.setter
    def loss(self, loss):
        self.wrapped_model.loss = loss
    
    @staticmethod
    def load_from_config_path(config_path):
        with open(config_path, 'rb') as f:
            config = json.load(f)
        if config["checkpoint_path"][-3:] == "pkl":
            with open(config["checkpoint_path"], 'rb') as f:
                return dill.load(f)
        with open(config["checkpoint_path"], 'rb') as f:
            cp = torch.load(f)
        plmodel = PLWrapper(wrapped_model=get_models.get_model(config), preprocess_data=PreprocessFunction(config["dynamic_preprocess"]), **cp["hyper_parameters"]) 
        plmodel.load_state_dict(cp["state_dict"])
        return plmodel
    # @property
    # def model(self):
    #     return self.wrapped_model.model
    # @model.setter
    # def model(self, model):
    #     self.wrapped_model.model = model

def get_lbfgs(params, *args, **kwargs):
    return torch.optim.LBFGS(params, lr=1, max_iter=1000)
    
OPTIM_DICT = {
    "adam": torch.optim.Adam,
    "lbfgs": torch.optim.LBFGS,
}

def get_model(config):
    lr = config["lr"] if "lr" in config.keys() else 3e-4
    lr = lr if config["trainer"] != "lbfgs" else 1
    optimizer = OPTIM_DICT[config["trainer"]]
    preprocess = PreprocessFunction(config["dynamic_preprocess"])
    return PLWrapper(get_models.get_model(config), lr=lr, optimizer=optimizer, preprocess_data=preprocess)

def sum_dict_list(dlist):
    dsum = {d: [] for d in dlist[0].keys()}
    for d in dlist:
        for k,v in d.items():
            dsum[k].append(v)
    return {k: torch.cat(v, dim=0) for k,v in dsum.items()}

def get_fix_dataloader(ds, inds, batch_size=1, num_workers=os.cpu_count()//2):
    sampler = BatchSampler(
        SubsetRandomSampler(inds),
        batch_size=batch_size,
        drop_last=True
    )
    dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=num_workers)
    return dl

class ArrayDataset(Dataset):
    def __init__(self, array):
        self.array = array
        self.length = len(array)
    def __len__(self):
        return self.length
    def __getitem__(self, inds):
        if inds is None:
            return sum_dict_list(self.array)
        elif isinstance(inds, int):
            return self.array[inds]
        elif isinstance(inds, slice):
            return sum_dict_list(self.array[inds])
        return sum_dict_list([self.array[i] for i in inds])
    def map(self, f):
        self.array = [f(x) for x in self.array]
        return self
    
class IterableDataloader():
    def __init__(self, iterable):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    
def to_device(x, device='cpu'):
    if isinstance(x, dict):
        return {k: to_device(v, device=device) for k,v in x.items()}
    else:
        return x.to(device)

def get_fix_dataloader_preload(ds, inds, batch_size=1, num_workers=os.cpu_count()//2, device='cpu'):
    dl = get_fix_dataloader(ds, inds, batch_size=batch_size, num_workers=num_workers)
    for x in dl:
        yield to_device(x, device=device)

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
    
class LRFinder(LearningRateFinder):
    def __init__(self, early_stop_threshold=None, num_training_steps=100, mode="exponential", **kwargs):
        super().__init__(early_stop_threshold=early_stop_threshold, num_training_steps=num_training_steps, mode=mode, **kwargs)
        self.best_val_loss = torch.inf
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            pl_module.update_lr(self.optimal_lr.suggestion())
        if  "val_loss" in trainer.logged_metrics and trainer.logged_metrics["val_loss"] >= self.best_val_loss:
            with print_off():
                self.lr_find(trainer, pl_module)
            if hasattr(pl_module, 'update_lr'):
                pl_module.update_lr(self.optimal_lr.suggestion())
            else:
                print('model does not have update_lr method. Please write one or ensure LR is changing... ')
        else:
            if "val_loss" in trainer.logged_metrics:
                self.best_val_loss = trainer.logged_metrics["val_loss"]

def get_dirnames(config):
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
