'''
    Script for replicating Jake's transient figures from scratch
    NOTE: 
    1. there are still some dependencies to NDN
    2. Ray is not used in this example
    3. This retrains and applies the shifter
'''
#%%
import os
import torch
import dill
from ray.air.checkpoint import Checkpoint
import matplotlib.pyplot as plt
import matplotlib
from models import CNNdense, ModelWrapper
from datasets.mitchell.pixel.utils import get_stim_list
from utils import train_loop_org, unpickle_data, seed_everything, ModelGenerator

# matplotlib.use('Agg')
seed = 0
seed_everything(420)

intended_device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
print('Intended device: ', intended_device)

%load_ext autoreload
%autoreload 2

#%%
'''
    User-Defined Parameters
'''
RUN_NAME = 'foundation_run' # Name of log dir.
nsamples_train=None #200000#236452
nsamples_val=None #56643

#%%
sesslist = list(get_stim_list().keys())
SESSION_NAME = '20200304'  # '20220610'
NBname = f'shifter_{SESSION_NAME}_{seed}'
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
from utils import memory_clear, train, get_datasets
from models.shifters import Shifter
memory_clear()

# dataset on cpu
train_data, val_data = unpickle_data(device=torch.device('cpu'))
train_dl, val_dl, _, _ = get_datasets(train_data, val_data, device=torch.device('cpu'), batch_size=1000)

#%% train shifter model        
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
    scaffold=[len(num_channels)-1], # output last layer
    reg_vals_feat={'l1': 0.01},
    reg_readout={'glocalx': 0.1})

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
