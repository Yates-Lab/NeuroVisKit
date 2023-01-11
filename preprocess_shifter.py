
'''
This script learns the shifter model for a particular session.
Run this before running preprocess.
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
batch_size = 1000
window_size = 35
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
    requested_stims=['Gabor', 'BackImage'],
    num_lags=num_lags,
    downsample_t=tdownsample,
    download=True,
    valid_eye_rad=valid_eye_rad,
    spike_sorting=spike_sorting,
    fixations_only=False,
    load_shifters=False,
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

FracDF_include = .2
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]

crop_window = ds.get_crop_window(inds=gab_inds, win_size=window_size, cids=cids, plot=True)
ds.crop_idx = crop_window

stas = ds.get_stas(inds=gab_inds, square=True)
cids = np.intersect1d(cids, np.where(~np.isnan(stas.std(dim=(0,1,2))))[0])

#%%
maxsamples = 197144
train_inds, val_inds = ds.get_train_indices(max_sample=int(0.85*maxsamples))

train_data = ds[train_inds]
train_data['stim'] = torch.flatten(train_data['stim'], start_dim=1)
val_data = ds[val_inds]
val_data['stim'] = torch.flatten(val_data['stim'], start_dim=1)

cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

input_dims = ds.dims + [ds.num_lags]

#%% Put dataset on GPU
from datasets import GenericDataset

dataset_device = torch.device('cpu') #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_ds = GenericDataset(train_data, device=train_device)
val_ds = GenericDataset(val_data, device=dataset_device) # we're okay with being slow

#%%
NUM_WORKERS = os.cpu_count()//2
from torch.utils.data import DataLoader
batch_size = 1000
if train_ds.device.type=='cuda':
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)

if val_ds.device.type=='cuda':
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
else:
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

# from utils import get_datasets, seed_everything
# train_dl, val_dl, _, _ = get_datasets(train_data, val_data, device=dataset_device, batch_size=batch_size)

#%% 
from models import CNNdense, Shifter
# import models.cnns as cnns
# from models.shifters import Shifter
# import torch.nn.functional as F
from copy import deepcopy
from NDNT.training import Trainer, EarlyStopping
from utils import initialize_gaussian_envelope

# def fit_cnn(input_dims, num_filters, filter_width, num_inh,
#     model_type='CNNdense',
#     name=None,
#     fit=False,
#     scaffold=None,
#     shifter=True,
#     early_stopping_patience=4,
#     window=None,
#     reg_core={'d2t': 1e-3, 'center': 1e-2, 'edge_t': .1},
#     reg_hidden={'glocalx': 1e-1, 'd2x': 1e-4, 'center':1e-3},
#     reg_readout={'l2': 1e-5},
#     reg_vals_feat={'l1':0.01}):


#     cr0 = cnns.__dict__[model_type](input_dims,
#             num_subunits=num_filters,
#             filter_width=filter_width,
#             num_inh=num_inh,
#             cids=cids,
#             bias=False,
#             scaffold=scaffold,
#             is_temporal=False,
#             batch_norm=True,
#             window=window,
#             norm_type=0,
#             reg_core=reg_core,
#             reg_hidden=reg_hidden,
#             reg_readout=reg_readout,
#             reg_vals_feat=reg_vals_feat,
#                         )
#             # modifiers = modifiers
#     # cr0.name = name

#     # cr0.bias.data = b0.bias.data.clone() #
#     cr0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,cids].mean(dim=0)) - 1)

#     w_centered = initialize_gaussian_envelope( cr0.core[0].get_weights(to_reshape=False), cr0.core[0].filter_dims)
#     cr0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)

#     cr0.prepare_regularization()

#     print("Custom initialization complete")
#     if shifter:
#         smod = Shifter(cr0, affine=True)
#     else:
#         smod = deepcopy(cr0)

#     optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)

#     earlystopping = EarlyStopping(patience=early_stopping_patience, verbose=False)

#     trainer = Trainer(optimizer=optimizer,
#         device = train_device,
#         dirpath = os.path.join(dirname, NBname, smod.name),
#         log_activations=False,
#         early_stopping=earlystopping,
#         verbose=2,
#         max_epochs=100)

#     # fit
#     trainer.fit(smod, train_dl, val_dl)

#     del trainer
#     torch.cuda.empty_cache()

#     return smod

def fit_shifter_model(cp_dir, affine=False, overwrite=False):
    from utils import train, memory_clear

    if affine:
        cp_dir = cp_dir + '_affine'
    
    if os.path.isdir(cp_dir) and not overwrite:
        fpath = os.path.join(cp_dir, 'model.pkl')
        if os.path.exists(fpath) and not overwrite:
            model = dill.load(open(fpath, 'rb'))
            model.to(dataset_device)
            val_loss_min = 0
            for data in val_dl:
                val_loss_min += model.validation_step(data)

            val_loss_min/=len(val_dl)    
            return model, val_loss_min.item() 

    os.makedirs(cp_dir, exist_ok=True)

    max_epochs = 150
    num_filters = [20, 20, 20, 20]
    filter_width = [11, 9, 7, 7]
    num_inh = [0]*len(num_filters)
    scaffold = [len(num_filters)-1]
    cr0 = CNNdense(input_dims,
            num_subunits=num_filters,
            filter_width=filter_width,
            num_inh=num_inh,
            cids=cids,
            bias=False,
            scaffold=scaffold,
            is_temporal=False,
            batch_norm=True,
            window='hamming',
            norm_type=0,
            reg_core=None,
            reg_hidden=None,
            reg_readout={'glocalx':1},
            reg_vals_feat={'l1':0.01},
                        )
        
    cr0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,cids].mean(dim=0)) - 1)
    w_centered = initialize_gaussian_envelope( cr0.core[0].get_weights(to_reshape=False), cr0.core[0].filter_dims)
    cr0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)
    cr0.prepare_regularization()

    smod = Shifter(cr0, affine=affine)

    optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)
    # val_loss_min = train(smod.to(train_device),
    #         train_dl,
    #         val_dl,
    #         optimizer=optimizer,
    #         max_epochs=max_epochs,
    #         verbose=2,
    #         checkpoint_path=cp_dir,
    #         device=train_device,
    #         patience=20)
    
    earlystopping = EarlyStopping(patience=3, verbose=False)

    trainer = Trainer(optimizer=optimizer,
        device = train_device,
        dirpath = os.path.join(dirname, NBname, smod.name),
        log_activations=False,
        early_stopping=earlystopping,
        verbose=2,
        max_epochs=max_epochs)

    # fit
    memory_clear()
    trainer.fit(smod, train_dl, val_dl)
    val_loss_min = deepcopy(trainer.val_loss_min)
    del trainer
    memory_clear()
    
    return smod, val_loss_min
    
# %%
from utils import seed_everything
seed = 66
NBname = f'shifter_{SESSION_NAME}_{seed}'
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
cp_dir = os.path.join(dirname, NBname)

seed_everything(seed)
mod0, loss0 = fit_shifter_model(cp_dir, affine=False, overwrite=False)

seed_everything(seed)
mod1, loss1 = fit_shifter_model(cp_dir, affine=True, overwrite=False)

# #%%

# # seed_everything(seed)
# num_filters = [20, 20, 20, 20]
# filter_width = [11, 9, 7, 7]
# num_inh = [0]*len(num_filters)
# scaffold = [len(num_filters)-1]
# # mod0, loss0 = fit_shifter_model(cp_dir, affine=True, overwrite=True)
# mod0 = fit_cnn(input_dims, num_filters, filter_width, num_inh,
#         fit=True,
#         model_type='CNNdense',
#         scaffold=scaffold,
#         window='hamming',
#         early_stopping_patience=15,
#         reg_core=None, #{'d2t': 1e-2, 'center': 1, 'glocalx': 1, 'edge_t': .1},
#         reg_hidden=None, #{'glocalx': 1e-1, 'center':1e-3},
#         reg_readout={'glocalx':1},
#         reg_vals_feat={'l1':0.01},
#         name='CNN3layerDense')

# data = ds[:100]
# mod0.training_step(data)

#%%
from copy import deepcopy
shifters = [mod0.shifter, mod1.shifter]
shifter = deepcopy(shifters[np.argmin([loss0, loss1])])

out = {'cids': cids,
    'shifter': shifter,
    'shifters': shifters,
    'vernum': [0,1],
    'valloss': [loss0, loss1],
    'numlags': num_lags,
    'tdownsample': tdownsample,
    'eyerad': valid_eye_rad,
    'input_dims': mod0.input_dims,
    'seed': seed}

fname = 'shifter_' + SESSION_NAME + '_' + ds.spike_sorting + '.p'
fpath = os.path.join(datadir,fname)

with open(fpath, 'wb') as f:
    dill.dump(out, f)

# # %%
# from models.utils.general import eval_model
# ll0 = eval_model(mod0, val_dl)
# ll1 = eval_model(mod1, val_dl)
# # %%
# %matplotlib inline
# plt.plot(ll0, ll1, '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.show()
# # %%
# mod1.model.core[0].plot_filters()
# # %%
# from datasets.mitchell.pixel.utils import plot_shifter
# _ = plot_shifter(mod0.shifter)
# _ = plot_shifter(mod1.shifter)
# # %%

# iix = (train_data['stimid']==0).flatten()
# y = (train_data['robs'][iix,:]*train_data['dfs'][iix,:])/train_data['dfs'][iix,:].sum(dim=0).T
# stas = (train_data['stim'][iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

# from models.utils import plot_stas
# %matplotlib inline
# _ =  plot_stas(stas.numpy())

# shift = mod0.shifter(train_data['eyepos'])
# shift[:,0] = shift[:,0] / input_dims[1] * 2
# shift[:,1] = shift[:,1] / input_dims[2] * 2
# stas0 = (mod0.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

# _ =  plot_stas(stas0.detach().numpy())

# shift = mod1.shifter(train_data['eyepos'])
# shift[:,0] = shift[:,0] / input_dims[1] * 2
# shift[:,1] = shift[:,1] / input_dims[2] * 2
# stas1 = (mod1.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

# _ =  plot_stas(stas1.detach().numpy())
# # %%
