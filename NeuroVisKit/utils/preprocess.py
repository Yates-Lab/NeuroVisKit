import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# def plot_sacta(ds):
#     sacta = ds.covariates['fixation_onset'].T@(ds.covariates['robs']*ds.covariates['dfs'])
#     sacta /= ds.covariates['fixation_onset'].sum(dim=0)[:,None]
#     plt.figure()
#     return plt.plot(sacta.detach())

# def get_ds_analysis(ds, WINDOW_SIZE=35, FracDF_include=0.2):
#     gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()
#     # nat_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['BackImage']['inds']))[0].tolist()

#     cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]

#     crop_window = ds.get_crop_window(inds=gab_inds, win_size=WINDOW_SIZE, cids=cids, plot=True)
#     ds.crop_idx = crop_window

#     stas = ds.get_stas(inds=gab_inds, square=True)
#     cids = np.intersect1d(cids, np.where(~np.isnan(stas.std(dim=(0,1,2))))[0])

#     return stas, cids, gab_inds

# def get_data_from_ds(ds, maxsamples=None, move_to_cpu=True):
#     train_inds, val_inds = ds.get_train_indices(max_sample=maxsamples)
#     train_data = ds[train_inds]
#     train_data['stim'] = torch.flatten(train_data['stim'], start_dim=1)
#     val_data = ds[val_inds]
#     val_data['stim'] = torch.flatten(val_data['stim'], start_dim=1)

#     if move_to_cpu:
#         for value in train_data.values():
#             value.to('cpu')
#         for value in val_data.values():
#             value.to('cpu')

#     return train_data, val_data, train_inds, val_inds

# def store_data(train_data, val_data, path=None):
#     path_names = ['data', 'train_dir', 'val_dir']
#     paths = {
#         'data': path,
#         'train_dir': os.path.join(path, 'train'),
#         'val_dir': os.path.join(path, 'val'),
#     }
#     for i in path_names:
#         if not os.path.exists(paths[i]):
#             os.makedirs(paths[i])
    
#     for key, val in train_data.items():
#         torch.save(val, os.path.join(paths['train_dir'], '%s.pt'%key))
#         del val
#     for key, val in val_data.items():
#         torch.save(val, os.path.join(paths['val_dir'], '%s.pt'%key))
#         del val
#     return paths

# def get_ds(Pixel, datadir, session_name, num_lags, spike_sorting='kilowf', load_shifters=False, valid_eye_rad=8, tdownsample=2):
#     ds = Pixel(datadir,
#         sess_list=[session_name],
#         requested_stims=['Gabor', 'BackImage'],
#         num_lags=num_lags,
#         downsample_t=tdownsample,
#         download=True,
#         valid_eye_rad=valid_eye_rad,
#         spike_sorting=spike_sorting,
#         fixations_only=False,
#         load_shifters=load_shifters,
#         # enforce_fixation_shift=False,
#         covariate_requests={
#             'fixation_onset': {'tent_ctrs': np.arange(-15, 60, 1)},
#             'frame_tent': {'ntents': 40},
#             'fixation_num': True}
#         )
#     print('calculating datafilters')
#     ds.compute_datafilters(
#         to_plot=False,
#         verbose=False,
#         frac_reject=0.25,
#         Lhole=20)
#     print('[%.2f] fraction of the data used' %ds.covariates['dfs'].mean().item())
#     return ds
