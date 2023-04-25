'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, memory_clear, get_datasets
from utils.train import get_trainer, get_loss, ZNorm
from utils.get_models import get_model
seed_everything(0)

run_name = 'convex' # Name of log dir.
session_name = '20200304'
nsamples_train=236452
nsamples_val=56643
batch_size=1000
overwrite = False
from_checkpoint = False
device = torch.device("cpu")
seed = 420

# Prepare helpers for training.
print('Device: ', device)
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
data_dir = os.path.join(dirname, 'sessions', session_name)
os.makedirs(checkpoint_dir, exist_ok=True)
    
with open(os.path.join(dirname, 'sessions', session_name, 'session.pkl'), 'rb') as f:
    session = dill.load(f)

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device, batch_size=batch_size)
#%%
# Train model.
from scnn.solvers import AL
import numpy as np
from scnn.regularizers import NeuronGL1
from scnn.optimize import optimize_model, sample_gate_vectors
from scnn.models import ConvexReLU
from scnn.metrics import Metrics
# %load_ext autoreload
# %autoreload 2
device = torch.device("cuda:0")
x_train = train_data["stim"].clone().flatten(1)
y_train = train_data["robs"].clone()
x_val = val_data["stim"].clone().flatten(1)
y_val = val_data["robs"].clone()

fsize = 100
for i in range(len(train_ds)-1, fsize-1, -1):
    y_train[i] = y_train[i-fsize:i].mean(0)
for i in range(len(val_ds)-1, fsize-1, -1):
    y_val[i] = y_val[i-fsize:i].mean(0)

# create convex reformulation
max_neurons = 1
cids = session['cids']
hidden_d = 100
d = len(cids)
G = sample_gate_vectors(420, x_train.shape[1], hidden_d)
model = ConvexReLU(G, d, bias=True)
# specify regularizer and solver
regularizer = NeuronGL1(lam=0.001)
solver = AL(model, tol=1e-6)
# choose metrics to collect during training
metrics = Metrics(model_loss=True,
                  train_accuracy=True,
                  test_accuracy=True,
                  neuron_sparsity=True)
# train model!
model, metrics = optimize_model(model,
                                solver,
                                metrics,
                                x_train[:10000],
                                y_train[:10000, cids],
                                x_val[:1000],
                                y_val[:1000, cids],
                                regularizer,
                                device="cuda:0",
                                unitize_data=False)

# training accuracy
# train_acc = np.sum(np.sign(model(x_train)) == y_train) / len(y_train)
with open(os.path.join(checkpoint_dir, 'model.pkl'), 'wb') as f:
    dill.dump(model, f)

#%%
# best_val_loss = trainer(model, train_dl, val_dl, checkpoint_dir, device, patience=30)

#save metadata
with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
    to_write = {
        'run_name': run_name,
        'session_name': session_name,
        'nsamples_train': nsamples_train,
        'nsamples_val': nsamples_val,
        'seed': seed,
        'device': str(device),
        # 'best_val_loss': best_val_loss,
    }
    f.write(json.dumps(to_write, indent=2))

# # %%