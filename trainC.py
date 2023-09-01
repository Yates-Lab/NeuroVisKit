'''
    Script for generating a nice fitting pipeline.
'''
#%%
#!%load_ext autoreload
#!%autoreload 2
import os, traceback, shutil
import numpy as np
import json
import torch
import dill
from _utils.utils import seed_everything, joinCWD
from utils.loss import get_loss
from utils.train import InMemoryContiguousDataset3
from utils.datasets import FixationMultiDataset
import torch.nn.functional as F
import torch.nn as nn
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils.lightning import PLWrapper, get_fix_dataloader
import utils.lightning as utils
from _utils.utils import isInteractive, get_opt_dict
from _utils.train import backupPreviousModel
from tqdm import tqdm
#%%
seed_everything(0)

config_defaults = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'CNNdense', # utils/get_models.py for more model options
    'trainer': 'adam', # utils/train.py for more trainer options
    'filters': [20, 20, 20, 20], # the config is fed into the model constructor
    'kernels': [11, 11, 11, 11],
    'preprocess': [],# this further preprocesses the data before training
    'dynamic_preprocess': [], # this preprocesses the data dynamically during training
    'load_preprocessed': False, # this loads the last preprocessed data from the session directory
    'override_output_NL': False, # this overrides the output nonlinearity of the model according to the loss,
    'pretrained_core': None,
    'defrost': False,
    'batch_size': 3,
    'seed': 420,
    'device': '1,',
    'session': '20200304C',
    'name': 'testC',
    'overwrite': False,
    'from_checkpoint': False,
    'lightning': True,
    'fast': False,
    'compile': False,
    'accumulate_batches': 1,
    'custom_models_path': None,
}

# Here we make sure that the script can be run from the command line.
if not isInteractive():
    loaded_config = get_opt_dict([
        ('n:', 'name='),
        ('s:', 'seed=', int),
        # ('c:', 'config=', json.loads),
        ('o', 'overwrite'),
        (None, 'from_checkpoint'),
        ('d:', 'device='),
        (None, 'session='),
        ('l:', 'loss='),
        ('m:', 'model='),
        ('t:', 'trainer='),
        ('p:', 'preprocess='),
        (None, 'override_NL'),
        (None, 'pretrained_core='),
        (None, 'defrost'),
        ('b:', 'batch_size=', int),
        (None, 'dynamic_preprocess='),
        (None, 'load_preprocessed'),
        ('f', 'fast'),
        ('c', 'compile'),
        ('a:', 'accumulate_batches=', int),
        ('r:', 'lr=', float),
        ('g:', 'custom_models_path='), # this is the path to the custom models file
    ], default=config_defaults)
    if config_defaults['from_checkpoint']:
        with open(joinCWD('data', 'models', config_defaults["name"], 'config.json'), 'r') as f:
            config = json.load(f)
            config.update(loaded_config)
    else:
        config = config_defaults
else:
    config = config_defaults
    # config["load_preprocessed"] = True
    config["name"] = "testB"
    config["overwrite"] = True
    config["session"] = "20200304B"
    config["model"] = "cnnc"
#%%
# Prepare helpers for training.
config["dirname"] = joinCWD('data') # this is the directory where the data is stored
dirs, config, session = utils.prepare_dirs(config)
# %%
# Load data.
ds_disk = FixationMultiDataset.load(dirs["ds_dir"])
ds_disk.use_blocks = True
ds = InMemoryContiguousDataset3(ds_disk)
if config['load_preprocessed']:
    print("Loading preprocessed data")
    with open(os.path.join(dirs["session_dir"], 'preprocessed.pkl'), 'rb') as f:
        ds = dill.load(f)
elif config['preprocess']:
    dl = get_fix_dataloader(ds, np.arange(len(ds)), batch_size=1)
    new_ds = []
    preprocess_func = utils.PreprocessFunction(config["preprocess"])
    for batch in tqdm(dl, desc='Preprocessing data'):
        new_ds.append(preprocess_func(batch))
    ds = utils.ArrayDataset(new_ds)
    with open(os.path.join(dirs["session_dir"], 'preprocessed.pkl'), 'wb') as f:
        dill.dump(ds, f)
if config["trainer"] == "lbfgs":
    train_dl = iter([ds[session["train_inds"]]])
    val_dl = iter([ds[session["val_inds"]]])
else:
    if config["device"] != "cpu":
        if hasattr(ds, 'preload'):
            print("Preloading")
            device = "cuda:"+config["device"].split(',')[0]
            ds.preload(device)
        else:
            device = None
        train_dl = get_fix_dataloader(ds, session["train_inds"], batch_size=config['batch_size'], device=device)
        val_dl = get_fix_dataloader(ds, session["val_inds"], batch_size=config['batch_size'], device=device)
    else:
        train_dl = get_fix_dataloader(ds, session["train_inds"], batch_size=config['batch_size'])
        val_dl = get_fix_dataloader(ds, session["val_inds"], batch_size=config['batch_size'])

#%%
# Load model and preprocess data.
model = utils.prepare_model(config, dirs)
model.loss, nonlinearity = get_loss(config)
if hasattr(model.loss, 'prepare_loss'):
    model.loss.prepare_loss(train_dl, cids=model.cids)
if config['override_output_NL']:
    model.model.output_NL = nonlinearity
if config['compile']:
    torch.compile(model, mode='reduce-overhead')
    
# model = model.to(torch.device("cuda:1"))
# opt = model.configure_optimizers()
# for i in tqdm(train_dl):
#     i["robs"] = i["robs"][35:]
#     i["dfs"] = i["dfs"][35:]
#     loss = model.wrapped_model.training_step(to_device(i, torch.device("cuda:1")))["loss"]
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
# quit()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min'),
]
trainer_args = {
    "callbacks": [
        *(callbacks if config["trainer"] != "lbfgs" else []),
        ModelCheckpoint(
            dirpath=dirs["checkpoint_dir"],
            filename='model',
            save_top_k=1,
            monitor="val_loss" if config["trainer"] != "lbfgs" else "train_loss",
            verbose=1,
            every_n_epochs=1,
            save_last=False
        ),
    ],
    "accelerator": "cpu" if config["device"] == 'cpu' else "gpu",
    "logger": TensorBoardLogger(dirs["checkpoint_dir"], version=0),
    "accumulate_grad_batches": config["accumulate_batches"],
}
if config["device"] != 'cpu':
    trainer_args["devices"] = config["device"]

best_val_loss, error = 'inf (failed run)', None
try:
    trainer = pl.Trainer(**trainer_args, default_root_dir=dirs["checkpoint_dir"], max_epochs=1000 if config["trainer"] != "lbfgs" else 1)
    trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)
    best_val_loss = str(trainer.validate(dataloaders=val_dl, ckpt_path='best'))
except (RuntimeError, KeyboardInterrupt, ValueError) as e:
    error = str(traceback.format_exc())
    print(error)
    if error != 'KeyboardInterrupt':
        print('Training failed. Saving model anyway (backing up previous model).')
        backupPreviousModel(dirs)
    
#save metadata
with open(dirs["config_path"], 'w') as f:
    to_write = {
        **config,
        'best_val_loss': best_val_loss,
        'error': error,
        'checkpoint_path': trainer.checkpoint_callback.best_model_path
    }
    for i in to_write.keys():
        if hasattr(to_write[i], "tolist"):
            to_write[i] = to_write[i].tolist()
    f.write(json.dumps(to_write, indent=2))

if not config["compile"]:
    #copy model from trainer.checkpoint_callback.best_model_path to dirs["model_path"]
    # shutil.copy(trainer.checkpoint_callback.best_model_path, dirs["model_path"])
    
    with open(dirs["model_path"], 'wb') as f:
        dill.dump(PLWrapper.load_from_config_path(dirs["config_path"]), f)
        print('Model pickled.')
# %%
