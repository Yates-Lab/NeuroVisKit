'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__, shutil, traceback
import numpy as np
import json, yaml
import torch
import dill
from utils.utils import seed_everything, unpickle_data, get_datasets, get_opt_dict, uneven_tqdm, reclass
from utils.loss import get_loss
from datasets.mitchell.pixel import FixationMultiDataset
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler, IterableDataset
import torch.nn.functional as F
import torch.nn as nn
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils.lightning import PLWrapper, get_fix_dataloader
import utils.lightning as utils
from NDNT.training.trainer import LBFGSTrainer
import logging
from utils.postprocess import eval_model_fast
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import moten
logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
torch.set_float32_matmul_precision("medium")
seed_everything(0)

config_defaults = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'gaborC', # utils/get_models.py for more model options
    'trainer': 'lbfgs', # utils/train.py for more trainer options
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
    'name': 'gallant',
    'overwrite': True,
    'from_checkpoint': False,
    'lightning': True,
    'fast': False,
    'compile': False,
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    loaded_opts = get_opt_dict([
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
    ])
    config_defaults.update(loaded_opts)
    if config_defaults['from_checkpoint']:
        with open(os.path.join(os.getcwd(), 'data', 'models', config_defaults["name"], 'config.json'), 'r') as f:
            config = json.load(f)
            # keys_to_remove = ['device', 'overwrite', 'session', 'loss', 'trainer', 'pretr']
            config.update(loaded_opts)
    else:
        config = config_defaults
else:
    config = config_defaults
    config["load_preprocessed"] = True
    config["name"] = "dynamic"
    config["overwrite"] = True
#%%
# Prepare helpers for training.
config["dirname"] = os.path.join(os.getcwd(), 'data') # this is the directory where the data is stored
dirs, config, session = utils.prepare_dirs(config)

ds_org = FixationMultiDataset.load(dirs["ds_dir"])
ds_org.use_blocks = True
setattr(ds_org, "_nsamples", None)
# ds_org = reclass(ds_org, FixationMultiDataset(config["session"], '/Data/stim_movies/'))

nlags = 1
dl = get_fix_dataloader(ds_org, ds_org.get_stim_indices(), batch_size=config["batch_size"])
new_ds = []
pyramid = moten.pyramids.MotionEnergyPyramid(
    stimulus_vhsize=(70, 70),
    stimulus_fps=240,
    temporal_frequencies=[0,2,4,8,16,32,64,120],
    spatial_frequencies=[0,2,4,8,16,35],
    spatial_directions=[0,45,90,135,180,225,270,315],
    sf_gauss_ratio=0.6,
    max_spatial_env=0.3,
    filter_spacing=3.5,
    tf_gauss_ratio=10.,
    max_temp_env=0.3,
    include_edges=True,
    spatial_phase_offset=0.0,
    filter_temporal_width=36,
)
for batch in uneven_tqdm(dl, total=ds_org.nsamples, desc='Preprocessing data', leave=False):
        batch["stim"] = torch.from_numpy(
            pyramid.project_stimulus(
                batch["stim"].squeeze(1).numpy()
            )
        )
        new_ds.append(batch)
ds = utils.ArrayDataset(new_ds)
with open('preprocessed1.pkl', "wb") as f:
    dill.dump(ds, f)
# # %%
# # Load data.
# ds = FixationMultiDataset.load(dirs["ds_dir"])
# ds.use_blocks = True
# prep_dir = os.path.join(dirs["session_dir"], 'preprocessed_gabor.pkl')
# if os.path.exists(prep_dir):
#     with open(prep_dir, 'rb') as f:
#         ds = dill.load(f)
# else:
#     dl = get_fix_dataloader(ds, ds.get_stim_indices(), batch_size=config["batch_size"])
#     new_ds = []
#     # pyramid = moten.pyramids.MotionEnergyPyramid((70, 70), 45)
#     pyramid = moten.pyramids.MotionEnergyPyramid(
#         stimulus_vhsize=(70, 70),
#         stimulus_fps=240,
#         temporal_frequencies=[0,2,4,8,16,32,64,120],
#         spatial_frequencies=[0,2,4,8,16,35],
#         spatial_directions=[0,45,90,135,180,225,270,315],
#         sf_gauss_ratio=0.6,
#         max_spatial_env=0.3,
#         filter_spacing=3.5,
#         tf_gauss_ratio=10.,
#         max_temp_env=0.3,
#         include_edges=True,
#         spatial_phase_offset=0.0,
#         filter_temporal_width=36,
#     )
    
#     for batch in tqdm(dl, desc='Preprocessing data', leave=False):
#         print(batch["stim"].shape)
#         batch["stim"] = torch.from_numpy(
#             pyramid.project_stimulus(
#                 batch["stim"].squeeze(1).numpy()
#             )
#         )
#         print(batch["stim"].shape)
#         new_ds.append(batch)
#     ds = utils.ArrayDataset(new_ds)
#     with open(prep_dir, "wb") as f:
#         dill.dump(ds, f)

# lag = 3
# mn, std = ds[:]["stim"].mean(0), ds[:]["stim"].std(0)
# def zscore(x):
#     x["stim"] = (x["stim"] - mn) / std
#     x["stim"] = x["stim"][:-lag or None]
#     x["robs"] = x["robs"][lag:]
#     x["dfs"] = x["dfs"][lag:]
#     return x
# ds.map(zscore)
# seed_everything(0)
# inds = np.random.permutation(len(ds))
# session["train_inds"] = inds[:int(len(ds)*0.85)]
# session["val_inds"] = inds[int(len(ds)*0.85):]

# if config["trainer"] == "lbfgs":
#     train_dl = utils.IterableDataloader([ds[session["train_inds"]]])
#     val_dl = utils.IterableDataloader([ds[session["val_inds"]]])
#     device = f"cuda:{config['device'][0]}" if config["device"] != 'cpu' else "cpu"
#     train_dl.iterable[0] = utils.to_device(train_dl.iterable[0], device=device)
#     val_dl.iterable[0] = utils.to_device(val_dl.iterable[0], device=device)
    
# else:
#     train_dl = get_fix_dataloader(ds, session["train_inds"], batch_size=1)
#     val_dl = get_fix_dataloader(ds, session["val_inds"], batch_size=1)
    
# if False:  
#     lag = 0
#     nfilters = ds[0]["stim"].shape[1]
#     w = math.ceil(math.sqrt(nfilters))
#     stas = torch.zeros((65, w*w, 1))
#     n = torch.zeros((65))
#     mn, std = ds[:]["stim"].mean(0), ds[:]["stim"].std(0)
#     for batch in ds:
#         stim, robs, dfs = batch["stim"], batch["robs"], batch["dfs"]
#         stim = (stim-mn)/std
#         stim = F.pad(stim, (w*w-nfilters, 0, 0, 0))
#         for i in range(len(robs)-lag):
#             weights = (robs[i+lag]*dfs[i+lag]).reshape(-1, 1, 1)
#             corr = stim[i:i+1].T.unsqueeze(0)
#             stas += corr*weights
#             n+=dfs[i]
        
#     stas = stas / n.reshape(65, 1, 1)
#     stas = stas.reshape(65, w, w, 1)

#     from models.utils.plotting import plot_sta_movie
#     plot_sta_movie(stas.permute(1, 2, 3, 0).numpy())

#     # mn, mx = stas.min(), stas.max()
#     nrows, ncols = 9, 9
#     plt.figure(figsize=(5*ncols, 5*nrows))
#     for i in range(65):
#         plt.subplot(nrows, ncols, i+1)
#         plt.imshow(stas[i], interpolation="none")#, vmin=mn, vmax=mx)
#     plt.figure(figsize=(5, 2*65))
#     for i in range(65):
#         plt.subplot(65, 1, i+1)
#         plt.plot(stas[i].reshape(-1)[w*w-nfilters:])
# #%%
# # Load model and preprocess data.
# model = utils.prepare_model(config, dirs)
# model.loss, nonlinearity = get_loss(config)
# if hasattr(model.loss, 'prepare_loss'):
#     model.loss.prepare_loss(train_dl, cids=model.cids)
# if hasattr(model.model, 'prepare_model'):
#     model.model.prepare_model(train_dl, cids=model.cids)
# if config['override_output_NL']:
#     model.model.output_NL = nonlinearity
# if config['compile']:
#     torch.compile(model, mode='reduce-overhead')
    
# # model = model.to(torch.device("cuda:1"))
# # opt = model.configure_optimizers()
# # for i in tqdm(train_dl):
# #     i["robs"] = i["robs"][35:]
# #     i["dfs"] = i["dfs"][35:]
# #     loss = model.wrapped_model.training_step(utils.to_device(i, torch.device("cuda:1")))["loss"]
# #     opt.zero_grad()
# #     loss.backward()
# #     opt.step()
# # quit()

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min'),
# ]
# if not config["fast"]:
#     callbacks.append(utils.LRFinder(num_training_steps=200))
# trainer_args = {
#     "callbacks": [
#         *(callbacks if config["trainer"] != "lbfgs" else []),
#         ModelCheckpoint(dirpath=dirs["checkpoint_dir"], filename='model', save_top_k=1, monitor="val_loss" if config["trainer"] != "lbfgs" else "train_loss", verbose=1, every_n_epochs=1),
#     ],
#     "accelerator": "cpu" if config["device"] == 'cpu' else "gpu",
#     "logger": TensorBoardLogger(dirs["checkpoint_dir"], version=0),
# }
# if config["device"] != 'cpu':
#     trainer_args["devices"] = config["device"]

# best_val_loss, error = 'inf (failed run)', None 
# try:
#     if config["trainer"] == "lbfgs":
#         try:
#             trainer = LBFGSTrainer(optimizer=model.configure_optimizers(), device=torch.device(device), dirpath=False)
#             trainer.fit(model.wrapped_model, train_loader=list(train_dl), val_loader=list(val_dl))
#             best_val_loss = trainer.val_loss_min
#         except KeyboardInterrupt:
#             pass
#         ev = eval_model_fast(model.to(val_dl.iterable[0]["stim"].device), val_dl.iterable[0])
#         print(ev)
#         print("average bits per spike: ", np.mean(ev))
#         print("max bits per spike: ", max(ev))
#     else:
#         trainer = pl.Trainer(**trainer_args, default_root_dir=dirs["checkpoint_dir"], max_epochs=1000 if config["trainer"] != "lbfgs" else 1)
#         trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)
#         best_val_loss = str(trainer.validate(dataloaders=val_dl, ckpt_path='best'))
# except (RuntimeError, KeyboardInterrupt, ValueError) as e:
#     error = str(traceback.format_exc())
#     print(error)
#     print('Training failed. Saving model anyway (backing up previous model).')
#     if os.path.exists(dirs["model_path"]):
#         shutil.copy(dirs["model_path"], dirs["model_path"][:-4] + '_backup.pkl')
#     if os.path.exists(dirs["config_path"]):
#         shutil.copy(dirs["config_path"], dirs["config_path"][:-5] + '_backup.json')
    
# #save metadata
# with open(dirs["config_path"], 'w') as f:
#     to_write = {
#         **config,
#         'best_val_loss': best_val_loss,
#         'error': error,
#         'checkpoint_path': trainer.checkpoint_callback.best_model_path if not config["trainer"] == "lbfgs" else dirs["model_path"],
#     }
#     for i in to_write.keys():
#         if hasattr(to_write[i], "tolist"):
#             to_write[i] = to_write[i].tolist()
#     f.write(json.dumps(to_write, indent=2))
# try:
#     with open(dirs["model_path"], 'wb') as f:
#         dill.dump(PLWrapper.load_from_config_path(dirs["config_path"]), f)
#         print('Model pickled.')
# except Exception as e:
#     print('Failed to pickle model.')
#     print(e)
# # %%
