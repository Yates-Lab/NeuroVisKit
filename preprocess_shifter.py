
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
SESSION_NAME = '20220610'
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
    spike_sorting='kilo',
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

#%%
from utils import get_datasets, seed_everything
train_dl, val_dl, _, _ = get_datasets(train_data, val_data, device=dataset_device, batch_size=batch_size)

#%% 
def fit_shifter_model(cp_dir, affine=False, overwrite=False):
    from models import CNNdense, Shifter
    from utils import train

    if affine:
        cp_dir = cp_dir + '_affine'
    
    if os.path.isdir(cp_dir):
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
    num_channels = [20, 20, 20, 20]
    filter_width = [11, 9, 7, 7]

    # base model
    c0 = CNNdense(input_dims=input_dims,
        cids=cids,
        num_subunits=num_channels,
        filter_width=filter_width,
        num_inh=[0]*len(num_channels),
        batch_norm=True,
        bias=False,
        window='hamming',
        padding='valid',
        scaffold=[len(num_channels)-1], # output last layer
        reg_core=None,#{'d2t':0.001, 'center':0.001},
        reg_hidden=None,#{'localx': .01},
        reg_vals_feat={'l1': 0.1},
        reg_readout={'glocalx': 1})
    
    from utils import initialize_gaussian_envelope
    w_centered = initialize_gaussian_envelope( c0.core[0].get_weights(to_reshape=False), c0.core[0].filter_dims)
    c0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)

    c0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,cids].mean(dim=0)) - 1)

    # shifter wrapper
    model = Shifter(c0, affine=affine)
    model.prepare_regularization()

    optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001)
    
    val_loss_min = np.nan
    from NDNT.training import Trainer, EarlyStopping, LBFGSTrainer

    earlystopping = EarlyStopping(patience=20, verbose=False)

    trainer = Trainer(optimizer=optimizer,
        device = train_device,
        dirpath = cp_dir,
        log_activations=False,
        early_stopping=earlystopping,
        verbose=2,
        max_epochs=max_epochs)

    # fit
    trainer.fit(model, train_dl, val_dl)

    # val_loss_min = train(model.to(train_device),
    #         train_dl,
    #         val_dl,
    #         optimizer=optimizer,
    #         max_epochs=max_epochs,
    #         verbose=2,
    #         checkpoint_path=cp_dir,
    #         device=train_device,
    #         patience=20)
    
    return model, val_loss_min
# %%
seed = 6
NBname = f'shifter_{SESSION_NAME}_{seed}'
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
cp_dir = os.path.join(dirname, NBname)

# seed_everything(seed)
mod0, loss0 = fit_shifter_model(cp_dir, affine=True, overwrite=True)

data = ds[:100]
mod0.training_step(data)

#%%
seed_everything(seed)
mod1, loss1 = fit_shifter_model(cp_dir, affine=True, overwrite=True)

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
    'seed': seed}

fname = 'shifter_' + SESSION_NAME + '_' + ds.spike_sorting + '.p'
fpath = os.path.join(datadir,fname)

with open(fpath, 'wb') as f:
    dill.dump(out, f)

# %%
from models.utils.general import eval_model
ll0 = eval_model(mod0, val_dl)
ll1 = eval_model(mod1, val_dl)
# %%
%matplotlib inline
plt.plot(ll0, ll1, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.show()
# %%
mod0.model.core[0].plot_filters()
# %%
from datasets.mitchell.pixel.utils import plot_shifter
_ = plot_shifter(mod0.shifter)
_ = plot_shifter(mod1.shifter)
# %%

iix = (train_data['stimid']==0).flatten()
y = (train_data['robs'][iix,:]*train_data['dfs'][iix,:])/train_data['dfs'][iix,:].sum(dim=0).T
stas = (train_data['stim'][iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

from models.utils import plot_stas
%matplotlib inline
_ =  plot_stas(stas.numpy())

shift = mod0.shifter(train_data['eyepos'])
shift[:,0] = shift[:,0] / input_dims[1] * 2
shift[:,1] = shift[:,1] / input_dims[2] * 2
stas0 = (mod0.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_ =  plot_stas(stas0.detach().numpy())

shift = mod1.shifter(train_data['eyepos'])
shift[:,0] = shift[:,0] / input_dims[1] * 2
shift[:,1] = shift[:,1] / input_dims[2] * 2
stas1 = (mod1.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_ =  plot_stas(stas1.detach().numpy())
# %%
