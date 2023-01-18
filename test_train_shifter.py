'''
    Script for replicating Jake's transient figures from scratch
    NOTE: 
    1. there are still dependencies to NDN
    2. Ray is not used in this example
    3. This retrains and applies the shifter
'''
#%%
import os
import torch
import dill
import matplotlib.pyplot as plt
from datasets.mitchell.pixel.utils import get_stim_list
from utils.utils import unpickle_data, seed_everything
# import matplotlib
# matplotlib.use('Agg')
seed = 2
seed_everything(seed)

train_device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
data_device = torch.device('cpu')
print('Training device: ', train_device)
print('Dataset device: ', data_device)

%load_ext autoreload
%autoreload 2

#%%
'''
    User-Defined Parameters
'''
RUN_NAME = 'test_run' # Name of log dir.
nsamples_train=None #200000#236452
nsamples_val=None #56643

#%%
sesslist = list(get_stim_list().keys())
SESSION_NAME = '20200304'  # '20220610'
assert SESSION_NAME in sesslist, "session name %s is not an available session" %SESSION_NAME

NBname = RUN_NAME + f'shifter_{SESSION_NAME}_{seed}'
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
cp_dir = os.path.join(dirname, NBname)
os.makedirs(cp_dir, exist_ok=True)

with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids, input_dims, mu, fix_inds = session['cids'], session['input_dims'], session['mu'], session['fix_inds']

fix_inds = fix_inds
NC = len(cids)


#%% get data loaders
from utils.utils import memory_clear, train, get_datasets
from models import Shifter, CNNdense
memory_clear()

#%%
# dataset on cpu
train_data, val_data = unpickle_data(device=torch.device('cpu'))
train_dl, val_dl, _, _ = get_datasets(train_data, val_data, device=torch.device('cpu'), batch_size=1000)

#%% train shifter model   
experiment_name = '_window_valid_long'
cp_dir = os.path.join(dirname, NBname + experiment_name)
os.makedirs(cp_dir, exist_ok=True)
input_dims = [1, 40, 40, 40]
max_epochs = 150
num_channels = [20, 30, 20, 20]
filter_width = [11, 9, 7, 5]

# base model
c0 = CNNdense(input_dims=input_dims,
    cids=cids,
    num_subunits=num_channels,
    filter_width=filter_width,
    num_inh=[0]*len(num_channels),
    batch_norm=True,
    window='hamming',
    padding='valid',
    scaffold=[len(num_channels)-1], # output last layer
    reg_core=None,
    reg_hidden=None,
    reg_vals_feat={'l1': 0.1},
    reg_readout={'glocalx': 1, 'l1': 0.1})

# shifter wrapper
mod0 = Shifter(c0, affine=True)
mod0.prepare_regularization()

optimizer = torch.optim.Adam(
        mod0.parameters(), lr=0.001)

val_loss_min = train(mod0.to(intended_device),
        train_dl,
        val_dl,
        optimizer=optimizer,
        max_epochs=max_epochs,
        verbose=2,
        checkpoint_path=cp_dir,
        device=intended_device,
        patience=20)


# %% stas raw
iix = (train_data['stimid']==0).flatten()
y = (train_data['robs'][iix,:]*train_data['dfs'][iix,:])/train_data['dfs'][iix,:].sum(dim=0).T
stas = (train_data['stim'][iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

#%%
from models.utils import plot_stas
%matplotlib inline
_ =  plot_stas(stas.numpy())

#%%
shift = mod0.shifter(train_data['eyepos'])
shift[:,0] = shift[:,0] / input_dims[1] * 2
shift[:,1] = shift[:,1] / input_dims[2] * 2
stas1 = (mod0.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

#%%
_ =  plot_stas(stas1.detach().numpy())

#%%
from datasets.mitchell.pixel.utils import plot_shifter
_ = plot_shifter(mod0.shifter)

# %%
from models.utils.general import eval_model
ll0 = eval_model(mod0, val_dl)
# %%
plt.plot(ll0)
# %%
mod0.model.core[0].plot_filters()
# %%

# %%
