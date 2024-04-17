#%%
'''
    Script for generating a nice fitting pipeline.
'''
#!%load_ext autoreload
#!%autoreload 2
from NeuroVisKit._utils import get_device_fancy, pl_device_format
import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from NeuroVisKit.utils.lightning import PLWrapper
from NeuroVisKit.utils.loss import Poisson
from NeuroVisKit.utils.models import PytorchWrapper
#%%
pl.seed_everything(0)
device = get_device_fancy("auto") #automatically select device with most available memory
checkpoint_dir = None # put checkpoint dir here
version = None #optional version number for logging
val_dl, train_dl = None, None # put dataloaders here
# Load model and preprocess data.
model = None #put model here
model = PytorchWrapper(model, loss=Poisson()) #loss defaults to Poisson
model = PLWrapper(model, lr=1e-3, optimizer=torch.optim.Adam) #optimizer defaults to Adam, LR defaults to 1e-3

trainer_args = {
    "callbacks": [
        EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min'),
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
    "logger": TensorBoardLogger(checkpoint_dir, version=0), #we recommend using wandb instead of tensorboard
}
if device != 'cpu':
    trainer_args["devices"] = pl_device_format(device) #if using GPU, specify device in PL format
    
trainer = pl.Trainer(**trainer_args, default_root_dir=checkpoint_dir, max_epochs=1000)
trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)