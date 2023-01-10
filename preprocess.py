
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
sessname = '20200304'
datadir = '/mnt/Data/Datasets/MitchellV1FreeViewing/stim_movies/' #'/Data/stim_movies/'
batch_size = 1000
window_size = 35
num_lags = 24
apply_shifter = False

#%%
# Process.
device = torch.device('cpu')
dtype = torch.float32
sesslist = list(get_stim_list().keys())
NBname = 'shifter_{}'.format(sessname)
cwd = os.getcwd()
maxsamples = None
path_names = ['data', 'train_dir', 'val_dir', 'log_dir']
paths = {
    'data': os.path.join(cwd, 'data'),
    'train_dir': os.path.join(cwd, 'data', 'train'),
    'val_dir': os.path.join(cwd, 'data', 'val'),
    'log_dir': os.path.join(cwd, 'data', 'tensorboard')
}
for i in path_names:
    if not os.path.exists(paths[i]):
        os.makedirs(paths[i])

valid_eye_rad = 8
ds = Pixel(datadir,
    sess_list=[sessname],
    requested_stims=['Gabor', 'BackImage'],
    num_lags=num_lags,
    downsample_t=2,
    download=True,
    valid_eye_rad=valid_eye_rad,
    spike_sorting='kilowf',
    fixations_only=False,
    load_shifters=apply_shifter,
    covariate_requests={
        'fixation_onset': {'tent_ctrs': np.arange(-15, 60, 1)},
        'frame_tent': {'ntents': 40}}
    )

print('calculating datafilters')
ds.compute_datafilters(
    to_plot=False,
    verbose=False,
    frac_reject=0.25,
    Lhole=20)

print('[%.2f] fraction of the data used' %ds.covariates['dfs'].mean().item())

gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()
nat_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['BackImage']['inds']))[0].tolist()

FracDF_include = .2
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]

crop_window = ds.get_crop_window(inds=gab_inds, win_size=window_size, cids=cids, plot=True)
ds.crop_idx = crop_window

stas = ds.get_stas(inds=gab_inds, square=True)
cids = np.intersect1d(cids, np.where(~np.isnan(stas.std(dim=(0,1,2))))[0])

stas = ds.get_stas(inds=gab_inds, square=True)
mu, bestlag = plot_stas(stas.detach().numpy())

sacta = ds.covariates['fixation_onset'].T@(ds.covariates['robs']*ds.covariates['dfs'])
sacta /= ds.covariates['fixation_onset'].sum(dim=0)[:,None]
plt.figure()
f = plt.plot(sacta.detach())

train_inds, val_inds = ds.get_train_indices()#max_sample=int(0.85*maxsamples))

#%%
train_data = ds[train_inds]
train_data['stim'] = torch.flatten(train_data['stim'], start_dim=1)
val_data = ds[val_inds]
val_data['stim'] = torch.flatten(val_data['stim'], start_dim=1)

for value in train_data.values():
    value.to('cpu')
for value in val_data.values():
    value.to('cpu')
    
print('New stim shape {}'.format(train_data['stim'].shape))
indsG = np.where(np.in1d(val_inds, gab_inds))[0]
indsN = np.where(~np.in1d(val_inds, gab_inds))[0]

print('{} training samples, {} validation samples, {} Gabor samples, {} Image samples'.format(len(train_inds), len(val_inds), len(indsG), len(indsN)))

cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

input_dims = ds.dims + [ds.num_lags]
mean_robs = ds.covariates['robs'][:,cids].mean(dim=0)

for key, val in train_data.items():
    torch.save(val, os.path.join(paths['train_dir'], '%s.pt'%key))
    del val
for key, val in val_data.items():
    torch.save(val, os.path.join(paths['val_dir'], '%s.pt'%key))
    del val

#%%
# get fixation indices.
fix_inds_org = ds.get_fixation_indices(index_valid=True)
sort_inds = np.argsort([max(i) if np.isin(i, train_inds).all() else np.inf for i in fix_inds_org])
fix_inds_sorted = [fix_inds_org[i] for i in sort_inds]
fix_inds_t = fix_inds_sorted[:len(train_inds)]

#%%
# save session.
session = {
    'fix_inds': fix_inds_t,
    'cids': cids,
    'mu': mu.copy(),
    'input_dims': input_dims
}
with open(os.path.join(paths['data'], 'session.pkl'), 'wb') as f:
    dill.dump(session, f)

#%%