#%%
import dill
import os
import torch
import numpy as np
import math
from copy import deepcopy
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import utils
from .utils import to_device
import torch.nn as nn
from NeuroVisKit.utils.mei import irfC, irf

#this makes sure if we move stuff around, users dont need to change imports.
from NeuroVisKit._utils.postprocess import *
from NeuroVisKit.utils.plotting import plot_grid, plot_model_conv, plot_split_grid

def get_zero_irfC(input_dims, model, neuron_inds=None, device='cpu'):
    rf = irfC(
        torch.zeros(input_dims, device=device),
        model,
        neuron_inds=neuron_inds,
    )
    return rf.detach().cpu()

class Evaluator(nn.Module):
    """Module that evaluates the log-likelihood of a model on a given dataset.
    This can be used for training.
    
    Instructions:
    - initialize with a list of neuron indices to evaluate.
    - call startDS to initialize null estimates (firing rate means).
    - 
    
    """
    def __init__(self, cids):
        super().__init__()
        self.cids = cids
        self.LLnull, self.LLsum, self.nspikes = 0, 0, 0
    def startDS(self, train_ds=None, means=None):
        """Initialize null estimates (firing rate means).

        Either provide a training dataset or a tensor of means.
        """
        if means is not None:
            self.register_buffer("mean_spikes", means)
        elif train_ds is not None:
            sum_spikes = (train_ds.covariates["robs"][:, self.cids] * train_ds.covariates["dfs"][:, self.cids]).sum(dim=0)
            self.register_buffer("mean_spikes", sum_spikes / train_ds.covariates["dfs"][:, self.cids].sum(dim=0))
        else:
            raise ValueError("Either train_ds or means must be provided.")
    def getLL(self, pred, batch):
        """Compute poisson log-likelihood of a batch of predictions.
        """
        poisson_ll = batch["robs"][:, self.cids] * torch.log(pred + 1e-8) - pred
        return (poisson_ll * batch["dfs"][:, self.cids]).sum(dim=0)
    def __call__(self, rpred, batch):
        """Evaluate model on a batch of data.
        Args:
            rpred (Tensor): predicted firing rates shaped (batch_size, num_neurons=len(cids)).
            batch (dict): dictionary of tensors with keys "dfs" and "robs".
        """
        llsum = self.getLL(rpred, batch).cpu()
        llnull = self.getLL(self.mean_spikes.to(rpred.device).expand(*rpred.shape), batch).cpu()
        self.LLnull = self.LLnull + llnull
        self.LLsum = self.LLsum + llsum
        self.nspikes = self.nspikes + (batch["dfs"][:, self.cids] * batch["robs"][:, self.cids]).sum(dim=0).cpu()
        del llsum, llnull, rpred, batch
    def closure(self):
        """Compute bits/spike for the neurons evaluated so far and reset counters.
        """
        bps = (self.LLsum - self.LLnull)/self.nspikes.clamp(1)/np.log(2)
        self.LLnull, self.LLsum, self.nspikes = 0, 0, 0
        return bps
    
def eval_model(model, val_dl, train_ds=None, means=None):
    '''
        Evaluate model on validation data.
        valid_data: either a dataloader or a dictionary of tensors.
        
        Provide either a training dataset or a tensor of training firing rate means.
    '''
    model.eval()
    with torch.no_grad():
        evaluator = Evaluator(model.cids)
        evaluator.startDS(train_ds=train_ds, means=means)
        for b in tqdm(val_dl, desc='Eval models'):
            evaluator(model(b), b)
        return evaluator.closure().detach().cpu().numpy()

def eval_model_summary(model, valid_dl, train_ds=None, means=None, topk=None, **kwargs):
    """Evaluate model on validation data and plot histogram of bits/spike.

    Provide either a training dataset or a tensor of training firing rate means.
    Optional topk argument to only plot the best topk neurons.
    """
    ev = eval_model(model, valid_dl, train_ds=train_ds, means=means)
    print(ev)
    if np.inf in ev or np.nan in ev:
        i = np.count_nonzero(np.isposinf(ev))
        ni = np.count_nonzero(np.isneginf(ev))
        print(f'Warning: {i} neurons have infinite/nan bits/spike, and {ni} neurons have ninf.')
        ev = ev[~np.isinf(ev) & ~np.isnan(ev)]
    # Creating histogram
    topk_ev = np.sort(ev)[-topk:] if topk is not None else ev
    _, ax = plt.subplots()
    ax.hist(topk_ev, bins=10)
    plt.axvline(x=np.max(topk_ev), color='r', linestyle='--')
    plt.axvline(x=np.min(topk_ev), color='r', linestyle='--')
    plt.xlabel("Bits/spike")
    plt.ylabel("Neuron count")
    plt.title("Model performance")
    # Show plot
    plt.show()
    return ev
# %%