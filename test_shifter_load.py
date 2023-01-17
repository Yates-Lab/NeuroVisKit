
'''
Test the output of the shifter once it's loaded by the dataset.
Use preprocess_shifter.py to train the shifter for the session in question

'''
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
spike_sorting = 'kilowf'
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
    spike_sorting=spike_sorting,
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
stas0 = ds.get_stas()
from models.utils import plot_stas
_ = plot_stas(stas0.detach().numpy())
# %%
ds = Pixel(datadir,
    sess_list=[SESSION_NAME],
    requested_stims=['Gabor'],
    num_lags=num_lags,
    downsample_t=tdownsample,
    download=True,
    valid_eye_rad=valid_eye_rad,
    spike_sorting=spike_sorting,
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
stas1 = ds.get_stas()
from models.utils import plot_stas

_, bestlag = plot_stas(stas1.detach().numpy())
cc = 0
# %% compare the two corrections

cc += 1
if cc >= stas1.shape[-1]:
    cc = 0
plt.figure()
plt.subplot(1,3,1)
plt.imshow(stas0[bestlag[cc], :,:,cc], interpolation='none')
plt.subplot(1,3,2)
plt.imshow(stas1[bestlag[cc], :,:,cc], interpolation='none')
plt.subplot(1,3,3)
plt.imshow(stas1[bestlag[cc], :,:,cc]-stas0[bestlag[cc], :,:,cc],  interpolation='none')


# %%
