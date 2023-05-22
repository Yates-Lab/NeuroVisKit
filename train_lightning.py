'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__, shutil
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, get_datasets
import utils.train as utils
from utils.loss import get_loss
from utils.get_models import get_model
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.postprocess import eval_model_summary
import lightning as pl
from lightning.pytorch.callbacks import LearningRateFinder, BatchSizeFinder, EarlyStopping
seed_everything(0)

class LitWrapper(pl.LightningModule):
    # This is a wrapper for NDNT model wrappers (ie a squared wrapper).
    def __init__(self, wrapped_model, train_ds, batch_size=1000, lr=1e-3, optimizer=torch.optim.Adam):
        super().__init__()
        # self.model = wrapped_model.model
        self.wrapped_model = wrapped_model
        self.opt = optimizer
        # self.train_dataset = train_ds
        self.batch_size = batch_size
        self.learning_rate = lr
        
    def forward(self, x):
        return self.wrapped_model(x)
    
    def configure_optimizers(self):
        return self.opt(self.wrapped_model.parameters(), lr=self.learning_rate)
    
    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count()//2)
    
    def training_step(self, train_batch, batch_idx):
        losses = self.wrapped_model.training_step(train_batch)
        self.log("train_loss", losses['train_loss'], prog_bar=True, on_epoch=True)
        if "reg_loss" in losses.keys():
            self.log("reg_loss", losses['reg_loss'], prog_bar=True, on_step=True)
        return losses['loss']
    
    def validation_step(self, val_batch, batch_idx):
        losses = self.wrapped_model.validation_step(val_batch)
        self.log("val_loss", losses["val_loss"], prog_bar=True, on_epoch=True)
        return losses["val_loss"]
    
    def on_validation_epoch_end(self):
        with torch.no_grad():
            with plt.ioff():
                ev = eval_model_summary(model, self.val_dataloader())
                self.log('summary', ev)
        return super().on_validation_epoch_end()

run_name = 'test' # Name of log dir.
session_name = '20200304'
nsamples_train=236452
nsamples_val=56643
overwrite = False
from_checkpoint = False
device = '1,' if torch.cuda.is_available() else 'cpu' # 'x' to train on first x gpus, 'x, y' to train on GPUs x and y, 'cpu' to train on cpu, 'auto' to train on all gpus
seed = 420
config = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'CNNdense', # utils/get_models.py for more model options
    'trainer': 'adam', # utils/train.py for more trainer options
    'filters': [20, 20, 20, 20], # the config is fed into the model constructor
    'kernels': [11, 11, 11, 11],
    'preprocess': 'binarize', # this further preprocesses the data before training
    'override_output_NL': True, # this overrides the output nonlinearity of the model according to the loss
    'lightning': True,
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite", "device=", "session=", "loss=", "model=", "trainer=", "preprocess=", "override_NL"]
    opts, args = getopt.getopt(argv,"n:s:oc:d:l:m:t:p:", opt_names)
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            run_name = arg
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-c", "--config"):
            config = config.update(json.loads(arg))
        elif opt == "--train":
            nsamples_train = int(arg)
        elif opt == "--val":
            nsamples_val = int(arg)
        elif opt in ("-o", "--overwrite"):
            overwrite = True
        elif opt in ("--from_checkpoint"):
            from_checkpoint = True
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("--session"):
            session_name = arg
        elif opt in ("-l", "--loss"):
            config["loss"] = arg
        elif opt in ("-m", "--model"):
            config["model"] = arg
        elif opt in ("-t", "--trainer"):
            config["trainer"] = arg
        elif opt in ("-p", "--preprocess"):
            config["preprocess"] = arg
        elif opt in ("--override_NL"):
            config["override_output_NL"] = True
        elif opt in ("--session"):
            session_name = arg
            
#%%
# Prepare helpers for training.
config_original = config.copy()
config['device'] = device
print('Device: ', device)
config['seed'] = seed
dirname = os.path.join(os.getcwd(), 'data')
light_dir = os.path.join(dirname, 'lightning')
checkpoint_dir = os.path.join(light_dir, run_name)
data_dir = os.path.join(dirname, 'sessions', session_name)
if os.path.exists(checkpoint_dir) and not overwrite and not from_checkpoint:
    print('Directory already exists. Exiting.')
    print('If you want to overwrite, use the -o flag.')
    sys.exit()
elif from_checkpoint:
    print('Loading from the best checkpoint has not been implemented for torch lightning.')
    raise NotImplementedError
elif overwrite:
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    else:
        print('Directory does not exist (did not overwrite).')

if not from_checkpoint:
    os.makedirs(checkpoint_dir)
    
with open(os.path.join(dirname, 'sessions', session_name, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
config.update({
    'cids': session['cids'],
    'input_dims': session['input_dims'],
    'mu': session['mu'],
})

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=torch.device('cpu'))

#%%
# Load model and preprocess data.
# if from_checkpoint:
#     model = LitWrapper.load_from_checkpoint(os.path.join(checkpoint_dir, 'model.ckpt'))
# else:
model = LitWrapper(get_model(config), train_ds, batch_size=64, lr=1e-3, optimizer=torch.optim.Adam)

model.wrapped_model.loss, nonlinearity = get_loss(config)
if config['override_output_NL']:
    model.wrapped_model.model.output_NL = nonlinearity

if config['preprocess'] == 'binarize': # Binarize the data to eliminate r > 1.
    inds2 = torch.where(train_data['robs'] > 1)[0]
    train_data['robs'][inds2 - 1] = 1
    train_data['robs'][inds2 + 1] = 1
    train_data['robs'][inds2] = 1
    inds2 = torch.where(val_data['robs'] > 1)[0]
    val_data['robs'][inds2 - 1] = 1
    val_data['robs'][inds2 + 1] = 1
    val_data['robs'][inds2] = 1
elif config['preprocess'] == 'smooth':
    train_data['robs'] = utils.smooth_robs(train_data['robs'], smoothN=10)
    val_data['robs'] = utils.smooth_robs(val_data['robs'], smoothN=10)
elif config['preprocess'] == 'zscore':
    train_data['robs'] = utils.zscore_robs(utils.smooth_robs(train_data['robs'], smoothN=10))
    val_data['robs'] = utils.zscore_robs(utils.smooth_robs(val_data['robs'], smoothN=10))

class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones
    def on_fit_start(self, *args, **kwargs):
        return
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)
class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, modulo=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulo = modulo
    def on_fit_start(self, *args, **kwargs):
        return
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.modulo == 0:
            self.lr_find(trainer, pl_module)
            
trainer_args = {
    "callbacks": [
        # EarlyStopping(monitor="val_loss", patience=30, verbose=True),
        # FineTuneBatchSizeFinder([]),
        FineTuneLearningRateFinder(modulo=100, early_stop_threshold=None)
    ]
}
if device == 'cpu':
    trainer_args.update({
        "accelerator": "cpu"
    })
else:
    trainer_args.update({
        "accelerator": "gpu",
        "devices": device
    })
    
trainer = pl.Trainer(**trainer_args, default_root_dir=checkpoint_dir, max_epochs=1000)
trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)
best_val_loss = trainer.validate(model, dataloaders=val_dl, ckpt_path=checkpoint_dir)["val_loss"].item()

#save metadata
with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
    to_write = {
        'run_name': run_name,
        'session_name': session_name,
        'nsamples_train': nsamples_train,
        'nsamples_val': nsamples_val,
        'seed': seed,
        'config': config_original,
        'device': str(device),
        'best_val_loss': best_val_loss,
    }
    f.write(json.dumps(to_write, indent=2))
with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as f:
    dill.dump(config_original, f)