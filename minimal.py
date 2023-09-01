'''
    Script for generating a nice fitting pipeline.
'''
#%%
#!%load_ext autoreload
#!%autoreload 2
from _utils.utils import seed_everything
import torch.nn.functional as F
import torch.nn as nn
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import utils.lightning as utils
from utils.models import PytorchWrapper
#%%
seed_everything(0)
device = 'cpu' # should be '0,' or '1,' or '0,1,' or 'cpu'
checkpoint_dir = None # put checkpoint dir here
version = None #optional version number for logging
accumulate_batches = 1 # number of batches to accumulate before backprop
val_dl, train_dl = None, None # put dataloaders here
# Load model and preprocess data.
model = None#put model here
#loss defaults to poisson if you dont set it
model = PytorchWrapper(model, loss=None, cids=None, bypass_preprocess=False) # bypass_preprocess allows dict to be passed straight to model
model = utils.PLWrapper(model, lr=1e-3, optimizer=None, preprocess_data=None) # preprocess_data is a function that dynamically preprocesses the data (see utils.lightning)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min'),
] # this breaks with LBFGS please remove
trainer_args = {
    "callbacks": [
        *callbacks,
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model',
            save_top_k=1,
            monitor="val_loss",
            verbose=1,
            every_n_epochs=1,
            save_last=False
        ),
    ],
    "accelerator": "cpu" if device == 'cpu' else "gpu",
    "logger": TensorBoardLogger(checkpoint_dir, version=0),
    "accumulate_grad_batches": accumulate_batches,
}
if device != 'cpu':
    trainer_args["devices"] = device
trainer = pl.Trainer(**trainer_args, default_root_dir=checkpoint_dir, max_epochs=1000)
trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)