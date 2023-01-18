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
from datasets.mitchell.pixel.utils import get_stim_list

#%%
'''
    User-Defined Parameters
'''
SESSION_NAME = '20191206'
spike_sorting = 'kilowf'
sesslist = list(get_stim_list().keys())
assert SESSION_NAME in sesslist, "session name %s is not an available session" %SESSION_NAME

datadir = '/mnt/Data/Datasets/MitchellV1FreeViewing/stim_movies/' #'/Data/stim_movies/'
batch_size = 1000
window_size = 35
num_lags = 24
seed = 1234
overwrite = False
retrain = False

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
# maxsamples = 197144
from NDNT.utils import get_max_samples
maxsamples = get_max_samples(ds, train_device)
train_inds, val_inds = ds.get_train_indices(max_sample=int(0.85*maxsamples))

train_data = ds[train_inds]
train_data['stim'] = torch.flatten(train_data['stim'], start_dim=1)
val_data = ds[val_inds]
val_data['stim'] = torch.flatten(val_data['stim'], start_dim=1)

cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

input_dims = ds.dims + [ds.num_lags]

#%% Put dataset on GPU
val_device = torch.device('cpu') # if you're cutting it close, put the validation set on the cpu
from utils.utils import get_datasets, seed_everything
train_dl, val_dl, _, _ = get_datasets(train_data, val_data, device=train_device, val_device=val_device, batch_size=batch_size)

#%% 
from models import CNNdense, Shifter
from copy import deepcopy
from NDNT.training import Trainer, EarlyStopping
from NDNT.utils import load_model
from utils.utils import initialize_gaussian_envelope


def fit_shifter_model(cp_dir, affine=False, overwrite=False):
    from utils.utils import memory_clear

    # manually name the model
    name = 'CNN_shifter'
    if affine:
        name = name + '_affine'
    
    # load best model if it already exists
    exists = os.path.isdir(os.path.join(cp_dir, name))
    if exists and not overwrite:
        try:
            smod = load_model(cp_dir, name)

            smod.to(dataset_device)
            val_loss_min = 0
            for data in val_dl:
                out = smod.validation_step(data)
                val_loss_min += out['loss'].item()

            val_loss_min/=len(val_dl)    
            return smod, val_loss_min
        except:
            pass

    os.makedirs(cp_dir, exist_ok=True)

    # parameters of architecture
    num_filters = [20, 20, 20, 20]
    filter_width = [11, 9, 7, 7]
    num_inh = [0]*len(num_filters)
    scaffold = [len(num_filters)-1]

    # build CNN
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
    
    # initialize parameters
    cr0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,cids].mean(dim=0)) - 1)
    w_centered = initialize_gaussian_envelope( cr0.core[0].get_weights(to_reshape=False), cr0.core[0].filter_dims)
    cr0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)
    
    # build regularization modules
    cr0.prepare_regularization()

    # wrap in a shifter network
    smod = Shifter(cr0, affine=affine)
    smod.name = name

    optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)
    
    # minimal early stopping patience is all we need here
    earlystopping = EarlyStopping(patience=3, verbose=False)

    trainer = Trainer(optimizer=optimizer,
        device = train_device,
        dirpath = os.path.join(cp_dir, smod.name),
        log_activations=False,
        early_stopping=earlystopping,
        verbose=2,
        max_epochs=100)

    # fit and cleanup memory
    memory_clear()
    trainer.fit(smod, train_dl, val_dl)
    val_loss_min = deepcopy(trainer.val_loss_min)
    del trainer
    memory_clear()
    
    return smod, val_loss_min
    
# %% fit shifter models
from utils.utils import seed_everything
NBname = f'shifter_{SESSION_NAME}_{seed}'
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
cp_dir = os.path.join(dirname, NBname)

# fit shifter with translation only
seed_everything(seed)
mod0, loss0 = fit_shifter_model(cp_dir, affine=False, overwrite=retrain)

# fit shifter with affine
seed_everything(seed)
mod1, loss1 = fit_shifter_model(cp_dir, affine=True, overwrite=retrain)

# %%
from models.utils import plot_stas, eval_model

ll0 = eval_model(mod0, val_dl)
ll1 = eval_model(mod1, val_dl)
# %%
%matplotlib inline
fig = plt.figure()
plt.plot(ll0, ll1, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.title("LL")
plt.xlabel("Translation shifter")
plt.ylabel("Affine Shifter")

plt.show()
# %%
mod1.model.core[0].plot_filters()
# %%
from datasets.mitchell.pixel.utils import plot_shifter
_,fig00 = plot_shifter(mod0.shifter, show=False)
_,fig01 = plot_shifter(mod1.shifter, show=False)
# %% plot STAs before and after shifting
iix = (train_data['stimid']==0).flatten()
y = (train_data['robs'][iix,:]*train_data['dfs'][iix,:])/train_data['dfs'][iix,:].sum(dim=0).T
stas = (train_data['stim'][iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_,_,fig02 =  plot_stas(stas.numpy(), title='no shift')

# do shift correction
shift = mod0.shifter(train_data['eyepos'])
stas0 = (mod0.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_,_,fig03 =  plot_stas(stas0.detach().numpy(), title='translation')

shift = mod1.shifter(train_data['eyepos'])
stas1 = (mod1.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_,_,fig04 =  plot_stas(stas1.detach().numpy(), title='affine')


#%%

model = mod0

def plotTransients(stimid=0, maxsamples=120):
    from tqdm import tqdm
    sacinds = np.where( (val_data['fixation_onset'][:,0] * (val_data['stimid'][:,0]-stimid)**2) > 1e-7)[0]
    nsac = len(sacinds)
    data = val_data

    print("Looping over %d saccades" %nsac)

    NC = len(model.cids)
    sta_true = np.nan*np.zeros((nsac, maxsamples, NC))
    sta_hat = np.nan*np.zeros((nsac, maxsamples, NC))

    for i in tqdm(range(len(sacinds)-1)):
        
        ii = sacinds[i]
        jj = sacinds[i+1]
        n = np.minimum(jj-ii, maxsamples)
        iix = np.arange(ii, ii+n)
        
        sample = {key: data[key][iix,:] for key in ['stim', 'robs', 'dfs', 'eyepos']}

        sta_hat[i,:n,:] = model(sample).detach().numpy()
        sta_true[i,:n,:] = sample['robs'][:,model.cids].detach().numpy()

    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    fig = plt.figure(figsize=(sx*2, sy*2))
    for cc in range(NC):
        
        plt.subplot(sx, sy, cc + 1)
        _ = plt.plot(np.nanmean(sta_true[:,:,cc],axis=0), 'k')
        _ = plt.plot(np.nanmean(sta_hat[:,:,cc],axis=0), 'r')
        plt.axis("off")
        plt.title(cc)

    plt.show()

    return sta_true, sta_hat, fig

sta_true, sta_hat, fig05 = plotTransients()
#%%
from matplotlib.backends.backend_pdf import PdfPages
filename = 'shifter_summary_%s_%d.pdf' %(SESSION_NAME, seed)
p = PdfPages(filename)
for fig in [fig, fig00, fig01, fig02, fig03, fig04, fig05]: 
    fig.savefig(p, format='pdf') 
      
p.close()  

#%% Save shifter output file

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

if not os.path.exists(fpath) or overwrite:
    with open(fpath, 'wb') as f:
        dill.dump(out, f)
# %%
