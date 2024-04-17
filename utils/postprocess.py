#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from NeuroVisKit.utils.evaluate import Evaluator

#this makes sure if we move stuff around, users dont need to change imports.
from NeuroVisKit._utils.postprocess import *
from NeuroVisKit.utils.plotting import plot_grid, plot_model_conv, plot_split_grid

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