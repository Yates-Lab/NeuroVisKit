'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, memory_clear, get_datasets
from utils.train import get_trainer, get_loss
from utils.get_models import get_model
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from tqdm import tqdm

seed_everything(0)

run_name = 'resor' # Name of log dir.
session_name = '20200304'
nsamples_train=236452
nsamples_val=56643
overwrite = False
from_checkpoint = False
device = torch.device('cpu')
seed = 420
config = {
    'loss': 'poisson',
    'model': 'resor',
    'trainer': 'adam',
    'preprocess': 'binarize',
}
            
#%%
# Prepare helpers for training.
config_original = config.copy()
config['device'] = device
print('Device: ', device)
config['seed'] = seed
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
data_dir = os.path.join(dirname, 'sessions', session_name)
os.makedirs(checkpoint_dir, exist_ok=True)
with open(os.path.join(dirname, 'sessions', session_name, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
config.update({
    'cids': session['cids'],
    'input_dims': session['input_dims'],
    'mu': session['mu'],
    'fix_inds': session['fix_inds'],
})

class RandomConvResBlock(nn.Module):
    def __init__(self, cin, cout, k=3, bias=True, residual=nn.Identity()):
        super().__init__()
        self.register_buffer('conv', nn.Conv2d(cin, cout, k, bias=bias, padding=k//2))
        self.register_buffer('residual', 0 if not residual else residual)
    def forward(self, x):
        return F.selu(self.conv(x) + self.residual(x))

def upResidual(cout=None):
    return lambda x: torch.cat([x, x], dim=1)[:, :cout]
def downResidual(cout=None):
    return lambda x: x[:, :cout]

class RandomResNet(nn.Module):
    def __init__(self, cout, depth=3, cmid=None, k=3, bias=True, seed=420):
        super().__init__()
        seed_everything(seed)
        cmid = cout if cmid is None else cmid
        step_up, step_down, core = [], [], []
        #binary step up from 1 to cmid
        for i in range(1, math.ceil(math.log2(cmid)) + 1):
            step_up.append(
                RandomConvResBlock(2**(i-1), 2**i, k, bias, residual=upResidual(2**i))
            )
        if cmid != cout:
            for i in range(math.ceil(math.log2(cmid)), math.ceil(math.log2(cout)), -1):
                step_up.append(
                    RandomConvResBlock(2**i, 2**(i-1), k, bias, residual=downResidual(2**(i-1)))
                )
            if cout != 2**math.ceil(math.log2(cout)):
                step_up.append(
                    RandomConvResBlock(2**math.ceil(math.log2(cout)), cout, k, bias, residual=downResidual(cout))
                )
        for i in range(depth - len(step_up) - len(step_down)):
            core.append(
                RandomConvResBlock(cmid, cmid, k, bias)
            )
        self.register_buffer(
            'model',
            nn.Sequential(
                *[*step_up, *core, *step_down]
            )
        )
    def forward(self, x):
        return self.model(x)

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device)

#%%
# Train model.
train_device = 'cuda:1'
batch_size = 1000
prefetch_factor = 10
cout = 1
depth = 264
cmid = 16

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, pin_memory_device=train_device, prefetch_factor=prefetch_factor)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, pin_memory_device=train_device, prefetch_factor=prefetch_factor)

model = RandomResNet(cout, depth=depth, cmid=cmid, k=3, bias=True, seed=420).to(train_device)
with torch.no_grad():
    train_x_projected = torch.empty_like(train_data['stim'])
    for i, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc='Projecting training data'):
        train_x_projected[i*batch_size:min((i+1)*batch_size, len(train_x_projected))] = model(batch['stim']).detach().cpu()
    torch.save(train_x_projected, 'train_x_projected/0.pt')
    del train_x_projected
    val_x_projected = torch.empty_like(val_data['stim'])
    for i, batch in tqdm(enumerate(val_dl), total=len(val_dl), desc='Projecting validation data'):
        val_x_projected[i*batch_size:min((i+1)*batch_size, len(val_x_projected))] = model(batch['stim']).detach().cpu()
    torch.save(val_x_projected, 'val_x_projected/0.pt')
# memory_clear()
# if from_checkpoint:
#     with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
#         model = dill.load(f).to(device)
# else:
#     model = get_model(config)

# model.loss, nonlinearity = get_loss(config)
# if config['override_output_NL']:
#     model.model.output_NL = nonlinearity

# def smooth_robs(x, smoothN=10):
#     smoothkernel = torch.ones((1, 1, smoothN, 1), device=device) / smoothN
#     out = F.conv2d(
#         F.pad(x, (0, 0, smoothN-1, 0)).unsqueeze(0).unsqueeze(0),
#         smoothkernel).squeeze(0).squeeze(0)  
#     assert len(x) == len(out)
#     return out
# def zscore_robs(x):
#     return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True)

# if config['preprocess'] == 'binarize':
#     inds2 = train_data['robs'] > 1
#     for i in inds2:
#         train_data['robs'][max(i-1, 0):min(i+1, len(train_ds))] = 1
#     inds2 = val_data['robs'] > 1
#     for i in inds2:
#         val_data['robs'][max(i-1, 0):min(i+1, len(val_ds))] = 1
# elif config['preprocess'] == 'smooth':
#     train_data['robs'] = smooth_robs(train_data['robs'], smoothN=10)
#     val_data['robs'] = smooth_robs(val_data['robs'], smoothN=10)
# elif config['preprocess'] == 'zscore':
#     train_data['robs'] = zscore_robs(smooth_robs(train_data['robs'], smoothN=10))
#     val_data['robs'] = zscore_robs(smooth_robs(val_data['robs'], smoothN=10))

# trainer = get_trainer(config)
# best_val_loss = trainer(model, train_dl, val_dl, checkpoint_dir, device)
# #save metadata
# with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
#     to_write = {
#         'run_name': run_name,
#         'session_name': session_name,
#         'nsamples_train': nsamples_train,
#         'nsamples_val': nsamples_val,
#         'seed': seed,
#         'config': config_original,
#         'device': str(device),
#         'best_val_loss': best_val_loss,
#     }
#     f.write(json.dumps(to_write, indent=2))