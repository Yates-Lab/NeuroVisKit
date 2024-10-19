import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal as signal
import torch.nn as nn

#this makes sure if we move stuff around, users dont need to change imports.
from NeuroVisKit._utils.utils import * 
from NeuroVisKit.utils.plotting import *

# import cv2

def event_triggered_op(covariate, events, range, inds=None, reduction=torch.mean):
    """Compute an event triggered operation (reverse correlation).

    Args:
        covariate (Tensor): (time, *any_shape)
        events (Tensor): (time, <optional channels>) binary signal or (n_events,) indices
        range (tuple): (-time before event, time after event) non exclusive (i.e. arange(range[0], range[1]))
        inds (Tensor): Indices along time to include in computation. Defaults to None (i.e. include all data).
        reduction (function): Function to reduce the data along time axis. Must accept dim as keyword argument.

    Returns:
        Tensor: (<optional channels>, range, *any_shape) where <optional channels> defaults to 1
    """
    cov_ndims, events_ndims = covariate.ndim-1, events.ndim-1
    if len(events) < covariate.shape[0]:
        events = torch.zeros(covariate.shape[0], dtype=torch.bool, device=events.device).scatter_(0, events, True)
    if covariate.ndim == 1:
        covariate = covariate.unsqueeze(1)
    assert len(events) > 0, "No events found"
    NT, sz = covariate.shape[0], list(covariate.shape[1:])
    NC = 1 if events.ndim == 1 else events.shape[1] # number of channels
    events = events.reshape(-1, NC)
    et = torch.zeros( [NC] + [range[1]-range[0]] + sz, dtype=torch.float32, device=covariate.device)
    inds = torch.arange(NT, device=et.device) if inds is None else inds.to(et.device)
    for i,lag in enumerate(torch.arange(range[0], range[1])):
        ix = inds[(inds+lag >= 0) & (inds+lag < NT)]  
        n_events = len(ix)   
        c, e = covariate[ix+lag,...], events[ix,...] # shapes (n_events, *any_shape), (n_events, <optional channels>)
        new_shapeC, new_shapeE = [n_events, 1, *sz], [n_events, NC, 1]
        et[:, i, ...] = reduction(c.reshape(new_shapeC) * e.reshape(new_shapeE), dim=0)
    if cov_ndims == 0:
        et = et.squeeze(-1)
    if events_ndims == 0:
        et = et.squeeze(0)
    return et

def r2(y, y_hat, dfs=None, dim=0):
    """Calculate the variance explained (r-squared) for pytorch tensors.
    """
    if dfs is None:
        ybar = torch.mean(y, axis=dim, keepdims=True)
        dfs = 1
    else:
        ybar = (y * dfs).sum(axis=dim, keepdim=True) / dfs.sum(axis=dim, keepdim=True)
        
    var_tot = torch.sum(dfs*(y - ybar)**2, axis=dim)
    var_res = torch.sum(dfs*(y - y_hat)**2, axis=dim)
    return 1 - var_res/var_tot 
    
def r2_numpy(y, y_hat, dfs=None, dim=0):
    """Calculate the variance explained (r-squared) for numpy arrays.
    """
    if dfs is None:
        ybar = np.mean(y, axis=dim, keepdims=True)
        dfs = 1
    else:
        y = np.where(dfs, y, 0)
        y_hat = np.where(dfs, y_hat, 0)
        ybar = (y*dfs).sum(axis=dim, keepdims=True) / dfs.sum(axis=dim, keepdims=True)
        
    var_tot = np.sum(dfs*(y - ybar)**2, axis=dim)
    var_res = np.sum(dfs*(y - y_hat)**2, axis=dim)
    return 1 - var_res/(var_tot + 1e-6)

def r2_dfs(y, yhat, dfs=None):
    ''' 
    calculate the variance explained (r-squared)
    Inputs:
        y: tensor of shape N x NC
        yhat: tensor of shape N x NC
        dfs: tensor of shape N x NC (1 for valid data points, 0 for invalid)

    Outputs:
        r2: tensor of shape NC
    '''
    if dfs is None:
        dfs = torch.ones(y.shape, device=y.device)
    ybar = (y * dfs).sum(dim=0) / dfs.sum(dim=0)
    sstot = torch.sum( ( y*dfs - ybar)**2, dim=0)
    ssres = torch.sum( (y*dfs - yhat)**2, dim=0)
    r2 = 1 - ssres/sstot
    return r2.detach().cpu()

def plot_transientsC_singleModel(model, val_dl, bins=(-40, 100), smooth=0, split=False, plot=True):
    device = extract_device(model)
    dims = model(next(iter(val_dl))).shape[1:]
    N = sum([len(batch['sac_on']) for batch in val_dl])
    sac_on = torch.zeros((N, 1), dtype=torch.bool, device=device)
    Y_hat = torch.zeros(N, *dims, dtype=torch.float32, device=device)
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_dl, desc="Preparing for transient computation."):
            b = len(batch['stim'])
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            sac_on[i:i+b] = batch['sac_on']
            Y_hat[i:i+b] = model(batch)
            i += b
    inds = torch.where(sac_on)[0].to(device)
    binsize = 1/240
    nsacs = len(inds)
    del sac_on
    transientY_hat = event_triggered_op(Y_hat, inds, bins, reduction=torch.sum).cpu().numpy()
    transientY_hat = transientY_hat/nsacs/binsize
    if smooth:
        #use savgol filter
        transientY_hat = signal.savgol_filter(transientY_hat, window_length=smooth, polyorder=1, axis=0)
    if not plot:
        return transientY_hat, None
    NC = transientY_hat.shape[-1]
    if split:
        assert NC % 2 == 0, "NC must be even for split"
    
    transientsA = transientY_hat[:, :NC//2]
    transientsB = transientY_hat[:, NC//2:]
    # alternate between rows of A and B
    nrows = int(np.ceil(np.sqrt(NC//2)))
    ncols = int(np.ceil(NC//2 / nrows))
    f = plt.figure(figsize=(3*ncols, 3*nrows))
    for cc in range(NC//2):
        plt.subplot(nrows, ncols, cc+1)
        plt.plot(transientsA[:, cc], 'red')
        plt.plot(transientsB[:, cc], 'blue')
        plt.xlim(*bins)
        plt.axis('tight')
        plt.axis('off')
        plt.title(cc)
    plt.tight_layout()
    return transientY_hat, f
        
def plot_transientsC_multiDS(ds1, ds2, bins=(-40, 100), filter=True, smooth=0):
    device = ds1[0]["stim"].device
    sac1 = torch.where(ds1.covariates['sac_on'])[0].to(device)
    sac2 = torch.where(ds2.covariates['sac_on'])[0].to(device)
    Y1 = ds1.covariates['robs']
    Y2 = ds2.covariates['robs']
    if filter:
        Y1 *= ds1.covariates['dfs']
        Y2 *= ds2.covariates['dfs']
    t1 = event_triggered_op(Y1, sac1, bins, reduction=torch.mean).cpu().numpy()
    t2 = event_triggered_op(Y2, sac2, bins, reduction=torch.mean).cpu().numpy()
    if smooth:
        #use savgol filter
        t1 = signal.savgol_filter(t1, window_length=smooth, polyorder=1, axis=0)
        t2 = signal.savgol_filter(t2, window_length=smooth, polyorder=1, axis=0)
    NC = t1.shape[-1]
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))
    f = plt.figure(figsize=(3*sx, 3*sy))
    for cc in range(NC):
        plt.subplot(sx,sy,cc+1)
        plt.plot(t1[:, cc], 'k')
        plt.plot(t2[:, cc], 'r')
        plt.xlim(*bins)
        plt.axis('tight')
        plt.axis('off')
        plt.title(cc)
    plt.tight_layout()
    return t1, t2, f

def calculate_transient_r2(model, val_dl, cids, bins=(-40, 100), filter=True, smooth=0):
    assert issubclass(type(val_dl), DataLoader), "val_dl must be a DataLoader"
    assert issubclass(type(model), nn.Module), "model must be a nn.Module"
    assert len(cids) > 0, "cids must be a list of channel ids"
    assert len(bins) == 2, "bins must be a tuple of length 2"
    # assert model_device(model) == dl_device(val_dl), "Model and data must be on same device"
    assert hasattr(val_dl.dataset, 'covariates'), "val_dl.dataset must have covariates"
    device = extract_device(model)
    N = sum([len(batch['sac_on']) for batch in val_dl])
    sac_on = torch.zeros((N, 1), dtype=torch.bool, device=device)
    Y = torch.zeros(N, len(cids), dtype=torch.float32, device=device)
    Y_hat = torch.zeros(N, len(cids), dtype=torch.float32, device=device)
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_dl, desc="Preparing for transient computation."):
            b = len(batch['stim'])
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            sac_on[i:i+b] = batch['sac_on']
            Y[i:i+b] = batch['robs'][:, cids]
            Y_hat[i:i+b] = model(batch)
            if filter:
                Y[i:i+b] *= batch['dfs'][:, cids]
            i += b
    inds = torch.where(sac_on)[0].to(device)
    del sac_on
    transientY = event_triggered_op(Y, inds, bins, reduction=torch.mean).cpu().numpy()
    del Y
    transientY_hat = event_triggered_op(Y_hat, inds, bins, reduction=torch.mean).cpu().numpy()
    if smooth:
        transientY = signal.savgol_filter(transientY, window_length=smooth, polyorder=1, axis=0)
    sortby = np.argsort(np.nanstd(transientY, axis=0))[::-1]
    r2_score = r2_numpy(transientY, transientY_hat, dim=0)
    return r2_score[sortby], sortby
    
def plot_transientsC_new(model, val_dl, cids, bins=(-40, 100), filter=True, smooth=0, r2_score=False, topk=None, cid_idx=None, plot=True, reduction=torch.sum):
    # assert issubclass(type(val_dl), DataLoader), "val_dl must be a DataLoader"
    assert issubclass(type(model), nn.Module), "model must be a nn.Module"
    assert len(cids) > 0, "cids must be a list of channel ids"
    assert len(bins) == 2, "bins must be a tuple of length 2"
    # assert model_device(model) == dl_device(val_dl), "Model and data must be on same device"
    # assert hasattr(val_dl.dataset, 'covariates'), "val_dl.dataset must have covariates"
    if cid_idx is None:
        cid_idx = torch.arange(len(cids))
    device = extract_device(model)
    N = sum([len(batch['sac_on']) for batch in val_dl])
    sac_on = torch.zeros((N, 1), dtype=torch.bool, device=device)
    Y = torch.zeros(N, len(cid_idx), dtype=torch.float32, device=device)
    Y_hat = torch.zeros(N, len(cid_idx), dtype=torch.float32, device=device)
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_dl, desc="Preparing for transient computation."):
            b = len(batch['stim'])
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            sac_on[i:i+b] = batch['sac_on']
            Y[i:i+b] = batch['robs'][:, cids][:, cid_idx]
            Y_hat[i:i+b] = model(batch)[:, cid_idx]
            if filter:
                Y[i:i+b] *= batch['dfs'][:, cids][:, cid_idx]
            i += b
    inds = torch.where(sac_on)[0].to(device)
    nsacs = len(inds)
    del sac_on
    transientY = event_triggered_op(Y, inds, bins, reduction=reduction).cpu().numpy()
    del Y
    transientY_hat = event_triggered_op(Y_hat, inds, bins, reduction=reduction).cpu().numpy()
    binsize = 1/240
    transientY = transientY/nsacs/binsize
    transientY_hat = transientY_hat/nsacs/binsize
    
    if smooth:
        #use savgol filter
        transientY = signal.savgol_filter(transientY, window_length=smooth, polyorder=1, axis=0)
        # transientY_hat = signal.savgol_filter(transientY_hat, window_length=smooth, polyorder=1, axis=0)
    
    if not plot:
        return transientY, transientY_hat, None
    NC = transientY.shape[-1]
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    f = plt.figure(figsize=(3*sx, 3*sy))
    for cc in range(NC):
        plt.subplot(sx,sy,cc+1)
        plt.plot(transientY[:, cc], 'k')
        plt.plot(transientY_hat[:, cc], 'r')
        plt.xlim(*bins)
        plt.axis('tight')
        plt.axis('off')
        plt.title(cids[cid_idx[cc]])
    plt.tight_layout()
    if r2_score:
        f2 = plt.figure()
        r2_score = r2_numpy(transientY, transientY_hat, dim=0)
        if topk is not None:
            r2_score = np.sort(r2_score[np.isfinite(r2_score)])[-topk:]
        plt.hist(r2_score)
        plt.xlabel("R2 score")
        plt.ylabel("Number of neurons")
        plt.title("R2 dist for transients smoothed with window length %d. Avg is %.2f" %(smooth, r2_score.mean()))
        return transientY, transientY_hat, [f2, f]
    return transientY, transientY_hat, f

def plot_transientsC_singleDS(val_dl, cids, bins=(-40, 100), filter=True, smooth=0, cid_idx=None, plot=True, reduction=torch.sum, device=None):
    # assert issubclass(type(val_dl), DataLoader), "val_dl must be a DataLoader"
    assert len(cids) > 0, "cids must be a list of channel ids"
    assert len(bins) == 2, "bins must be a tuple of length 2"
    if device is None:
        device = next(iter(val_dl))['stim'].device
    # assert model_device(model) == dl_device(val_dl), "Model and data must be on same device"
    # assert hasattr(val_dl.dataset, 'covariates'), "val_dl.dataset must have covariates"
    if cid_idx is None:
        cid_idx = torch.arange(len(cids))
    N = sum([len(batch['sac_on']) for batch in val_dl])
    sac_on = torch.zeros((N, 1), dtype=torch.bool, device=device)
    Y = torch.zeros(N, len(cid_idx), dtype=torch.float32, device=device)
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_dl, desc="Preparing for transient computation."):
            b = len(batch['stim'])
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            sac_on[i:i+b] = batch['sac_on']
            Y[i:i+b] = batch['robs'][:, cids][:, cid_idx]
            if filter:
                Y[i:i+b] *= batch['dfs'][:, cids][:, cid_idx]
            i += b
    inds = torch.where(sac_on)[0].to(device)
    nsacs = len(inds)
    del sac_on
    transientY = event_triggered_op(Y, inds, bins, reduction=reduction).cpu().numpy()
    del Y
    binsize = 1/240
    transientY = transientY/nsacs/binsize
    if smooth:
        transientY = signal.savgol_filter(transientY, window_length=smooth, polyorder=1, axis=0)
    if plot:
        x = np.arange(bins[0], bins[1])*binsize*1000
        def plotter(y):
            plt.plot(x, y)
        plot_square_grid1d(transientY.T[:, :, None], plotter=plotter, width=2.5)
    return transientY