
#%% Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import dill
from datasets.mitchell.pixel import Pixel
from models.utils import plot_stas
from datasets.mitchell.pixel.utils import get_stim_list

#%%
'''
    User-Defined Parameters
'''
SESSION_NAME = '20200304'
sesslist = list(get_stim_list().keys())
assert SESSION_NAME in sesslist, "session name %s is not an available session" %SESSION_NAME

datadir = '/mnt/Data/Datasets/MitchellV1FreeViewing/stim_movies/' #'/Data/stim_movies/'
num_lags = 24
seed = 0

#%%
# Process.
train_device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_device = torch.device('cpu')
dtype = torch.float32
sesslist = list(get_stim_list().keys())
NBname = 'shifter_{}'.format(SESSION_NAME)
cwd = os.getcwd()

valid_eye_rad = 8
tdownsample = 2
ds = Pixel(datadir,
    sess_list=[SESSION_NAME],
    requested_stims=['Gabor'],
    num_lags=num_lags,
    downsample_t=tdownsample,
    download=True,
    valid_eye_rad=valid_eye_rad,
    spike_sorting='kilowf',
    fixations_only=False,
    load_shifters=True,
    enforce_fixation_shift=True,
    )

print('calculating datafilters')
ds.compute_datafilters(
    to_plot=False,
    verbose=False,
    frac_reject=0.25,
    Lhole=20)

print('[%.2f] fraction of the data used' %ds.covariates['dfs'].mean().item())

gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()

FracDF_include = .2
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
ds.crop_idx = [5,ds.dims[1]-5,5,ds.dims[2]-5]

%matplotlib inline
stas = ds.get_stas()
from models.utils import plot_stas
_ = plot_stas(stas.detach().numpy())
# %%
ds = Pixel(datadir,
    sess_list=[SESSION_NAME],
    requested_stims=['Gabor'],
    num_lags=num_lags,
    downsample_t=tdownsample,
    download=True,
    valid_eye_rad=valid_eye_rad,
    spike_sorting='kilowf',
    fixations_only=False,
    load_shifters=True,
    enforce_fixation_shift=False,
    )

print('calculating datafilters')
ds.compute_datafilters(
    to_plot=False,
    verbose=False,
    frac_reject=0.25,
    Lhole=20)

print('[%.2f] fraction of the data used' %ds.covariates['dfs'].mean().item())

gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()

FracDF_include = .2
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
ds.crop_idx = [5,ds.dims[1]-5,5,ds.dims[2]-5]

%matplotlib inline
stas = ds.get_stas()
from models.utils import plot_stas
_ = plot_stas(stas.detach().numpy())
# %%
