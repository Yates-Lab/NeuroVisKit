'''
    For now mainly focusing on integrated gradients.
'''
#%%
import torch
import numpy as np
import dill
import os
from copy import deepcopy
from utils.utils import unpickle_data, get_datasets, eval_model
import matplotlib.pyplot as plt
device = torch.device("cpu") #torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
nsamples_train=10000
nsamples_val=56643
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids = session['cids']

def eval():
    train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
    train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device, batch_size=1)

    evals = []
    for i in range(100):
        print(f'Started {i}')
        folderName = f'checkpoint_{i}'# 'goodmodel_20'
        with open(os.path.join(dirname, folderName, 'model.pkl'), 'rb') as f:
            model = dill.load(f)
        model.to(device)
        evals.append(eval_model(model, val_dl))
        print(evals[-1])

    dill.dump(evals, open(os.path.join(dirname, 'evals.pkl'), 'wb'))

def get_evals():
    evals = dill.load(open(os.path.join(dirname, 'evals.pkl'), 'rb'))
    evals = np.array(evals)
    mxs = evals.max(1)
    mns = evals.min(1)
    rngs = mxs - mns
    mds = np.median(evals, axis=1)
    
    means = evals.mean(1)
    plt.figure(figsize=(10, 10))
    plt.suptitle('Distribution of Model Over Random Inits')

    # plt.subplot(2, 2, 1)
    # plt.hist(mns, density=True)
    # plt.title('Min Score Dist')
    # plt.ylabel('Density')
    # plt.xlabel('Min Score')

    plt.subplot(2, 2, 1)
    plt.hist(rngs, density=True)
    plt.title('Score Range Dist (max - min)')
    plt.ylabel('Density')
    plt.xlabel('Score Range')

    plt.subplot(2, 2, 2)
    plt.hist(mxs, density=True)
    plt.title('Max Score Dist')
    plt.ylabel('Density')
    plt.xlabel('Max Score')

    plt.subplot(2, 2, 3)
    plt.hist(means, density=True)
    plt.title('Mean Score Dist')
    plt.ylabel('Density')
    plt.xlabel('Mean Score')
    
    plt.subplot(2, 2, 4)
    plt.hist(mds, density=True)
    plt.title('Median Score Dist')
    plt.ylabel('Density')
    plt.xlabel('Median Score')

    plt.tight_layout()

    bins = np.linspace(-0.4, 1, 15)
    plt.figure(figsize=(10, 5))
    plt.suptitle('Distribution of Scores for Each Model')
    
    plt.subplot(1, 2, 1)
    for i in range(100):
        plt.hist(evals[i], density=True, alpha=0.1, color='black', bins=bins)
    plt.title('Predetermined Bins')
    plt.ylabel('Density')
    plt.xlabel('Score')

    plt.subplot(1, 2, 2)
    for i in range(100):
        plt.hist(evals[i], density=True, alpha=0.1, color='black')
    plt.title('Fuzzy')
    plt.ylabel('Density')
    plt.xlabel('Score')

    return evals
# %%
