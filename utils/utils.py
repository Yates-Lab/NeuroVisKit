import os, sys, getopt
import gc
import time
import random
from copy import deepcopy
import numpy as np
import torch
import dill
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from NeuroVisKit.utils.datasets.generic import GenericDataset
from tqdm import tqdm
from functools import reduce
from IPython.display import Video
import imageio
from NeuroVisKit._utils.utils import memory_clear
import torch.nn as nn
import torch.nn.functional as F
# import cv2

class split_nonlinearity(nn.Module):
    def __init__(self, f=nn.Softplus()):
        super().__init__()
        self.f = f
    def forward(self, x):
        return torch.cat([self.f(x), self.f(-x)], dim=1)
    
def pad_causal(x, layer, kdims=None):
    """
    Pad a tensor for a causal convolution.
    x is a tensor
    layer is a convolutional layer
    kdims is the dimensions of the kernel that should be causal
    """
    kdims = kdims if kdims is not None else np.arange(len(layer.kernel_size))
    kdims = [kdims] if type(kdims) is int else kdims
    pad_array = []
    for i in np.arange(len(layer.kernel_size))[::-1]:
        pad_array += [layer.kernel_size[i]-1, 0] if i in kdims else [0, 0]
    return F.pad(x, pad_array)

def KaimingTensor(shape, out_dim=-1, *args, **kwargs):
    t = torch.empty(shape, *args, **kwargs)
    nn.init.kaiming_uniform_(t, a=1/shape[out_dim])
    return t

def getattr_deep(obj, attr):
    return reduce(getattr, attr.split('.'), obj)

class TimeLogger():
    def __init__(self):
        self.timer = time.time()
        self.accumulated = 0
    def reset(self):
        self.accumulated += time.time() - self.timer
        self.timer = time.time()
    def log(self, msg):
        print(f'{msg} {(time.time() - self.timer):.1f}s')
        self.accumulated += time.time() - self.timer
        self.timer = time.time()
    def closure(self):
        self.reset()
        m, s = int(self.accumulated//60), self.accumulated%60
        print(f'Total run took {m}m {s}s')
        self.accumulated = 0
        
def to_device(x, device='cpu'):
    if torch.is_tensor(x):
        return x.to(device) if x.device != device else x
    elif isinstance(x, dict):
        return {k: to_device(v, device=device) for k,v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device=device) for v in x]
    return x
        
def plot_stim(stim, fig=None, title=None, subplot_shape=(1, 1)):
    if fig is None:
        plt.figure()
    if title is not None:
        plt.title(title)
    c = int(np.ceil(np.sqrt(stim.shape[-1])))
    r = int(np.ceil(stim.shape[-1] / c))
    for i in range(stim.shape[-1]):
        ind = (i%c) + (i//c)*c*subplot_shape[1]
        plt.subplot(r*subplot_shape[0],c*subplot_shape[1],ind+1)
        plt.imshow(stim[..., i], vmin=stim.min(), vmax=stim.max())
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    plt.tight_layout()
    return fig

def initialize_gaussian_envelope( ws, w_shape):
    """
    This assumes a set of filters is passed in, and windows by Gaussian along each non-singleton dimension
    ws is all filters (ndims x nfilters)
    wshape is individual filter shape
    """
    ndims, nfilt = ws.shape
    assert np.prod(w_shape) == ndims
    wx = np.reshape(deepcopy(ws), w_shape + [nfilt])
    for dd in range(1,len(w_shape)):
        if w_shape[dd] > 1:
            L = w_shape[dd]
            if dd == len(w_shape)-1:
                genv = np.exp(-(np.arange(L))**2/(2*(L/6)**2))
            else:
                genv = np.exp(-(np.arange(L)-L/2)**2/(2*(L/6)**2))

            if dd == 0:
                wx = np.einsum('abcde, a->abcde', wx, genv)
            elif dd == 1:
                wx = np.einsum('abcde, b->abcde', wx, genv)
            elif dd == 2:
                wx = np.einsum('abcde, c->abcde', wx, genv)
            else:
                wx = np.einsum('abcde, d->abcde', wx, genv)
    return np.reshape(wx, [-1, nfilt])

def get_datasets(train_data, val_data, device=None, val_device=None, batch_size=1000, force_shuffle=False, shuffle=True):
    '''
        Get datasets from data files.
    '''
    # train_ds = GenericDataset(train_data, device)
    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_ds = GenericDataset(val_data, device)
    # val_dl = DataLoader(val_ds, batch_size=batch_size)
    train_ds = GenericDataset(train_data, device=device)
    if val_device is None:
        val_device = device
    val_ds = GenericDataset(val_data, device=val_device) # we're okay with being slow
    train_dl = get_dataloader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_dl = get_dataloader(val_ds, batch_size=batch_size, shuffle=False or force_shuffle)
    return train_dl, val_dl, train_ds, val_ds

def get_dataloader(dataset, batch_size=1000, shuffle=True):
    if dataset.device.type=='cuda':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count()//2)

def unpickle_data(nsamples_train_limit=None, nsamples_val_limit=None, device="cpu", path=None):
    '''
        Get training and validation data from pickled files and place on device.
    '''
    path = os.path.join(os.getcwd(), 'data') if path is None else path
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    train_data_local, val_data_local = {}, {}
    num_samples = [None, None]
    for file in os.listdir(train_path):
        loaded = torch.load(os.path.join(train_path, file), map_location=device)
        num_samples[0] = loaded.shape[0]
        train_data_local[file[:-3]
                         ] = loaded[:nsamples_train_limit].clone()
        del loaded
        memory_clear()
    for file in os.listdir(val_path):
        loaded = torch.load(os.path.join(val_path, file), map_location=device)
        num_samples[1] = loaded.shape[0]
        val_data_local[file[:-3]
                       ] = loaded[:nsamples_val_limit].clone()
        del loaded
        memory_clear()
    print(f'Loaded {nsamples_train_limit} training samples and {nsamples_val_limit} validation samples')
    print(f'Out of {num_samples[0]} training samples and {num_samples[1]} validation samples')
    return train_data_local, val_data_local

def load_model(checkpoint_path, model):
    '''
        Load model from checkpoint of trainer, if using state dict.
    '''
    ckpt = dill.load(open(os.path.join(checkpoint_path, 'state.pkl'), 'rb'))
    model.load_state_dict(ckpt['net'])
    epoch = ckpt['epoch']
    print(f"Loaded model from checkpoint. {epoch} epochs trained.")
    return model

def plot_transients(model, val_data, stimid=0, maxsamples=120, device=None):
    if device is not None:
        model = model.to(device)
        # model.model = model.model.to(device)
        for key in ['stim', 'robs', 'dfs', 'eyepos', 'fixation_onset', 'stimid']:
            val_data[key] = val_data[key].to(device)
        
    sacinds = torch.where( (val_data['fixation_onset'][:,0] * (val_data['stimid'][:,0]-stimid)**2) > 1e-7)[0]
    nsac = len(sacinds)
    data = val_data

    print("Looping over %d saccades" %nsac)

    NC = len(model.cids)
    sta_true = torch.nan*torch.zeros((nsac, maxsamples, NC))
    sta_hat = torch.nan*torch.zeros((nsac, maxsamples, NC))

    for i in tqdm(range(len(sacinds)-1)):
        
        ii = sacinds[i]
        jj = sacinds[i+1]
        n = min(jj-ii, maxsamples)
        iix = torch.arange(ii, ii+n)
        
        sample = {key: data[key][iix,:] for key in ['stim', 'robs', 'dfs', 'eyepos']}

        sta_hat[i,:n,:] = model(sample)
        sta_true[i,:n,:] = sample['robs'][:,model.cids]

    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    fig = plt.figure(figsize=(sx*2, sy*2))
    for cc in range(NC):
        
        plt.subplot(sx, sy, cc + 1)
        _ = plt.plot(torch.nanmean(sta_true[:,:,cc],axis=0).cpu().detach(), 'k')
        _ = plt.plot(torch.nanmean(sta_hat[:,:,cc],axis=0).cpu().detach(), 'r')
        plt.axis("off")
        plt.title(cc)

    plt.show()

    return sta_true, sta_hat, fig

def dot_along_axis_f(axis=-1):
    return torch.vmap(lambda x, y: torch.dot(x, y), in_dims=(axis, axis))

def event_triggered_op(covariate, events, range, inds=None, reduction=torch.mean):
    # covariate is (time, *any_shape)
    # events is (time, <optional channels>) binary signal or (n_events,) indices
    # range is (-time before event, time after event) non exclusive (i.e. arange(range[0], range[1]))
    # reduction is a function that takes in dim as keyword argument
    # returns (<optional channels>, range, *any_shape) where <optional channels> defaults to 1
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
    inds = torch.arange(NT, device=et.device) if inds is None else inds
    for i,lag in enumerate(torch.arange(range[0], range[1])):
        # print('Computing ETO for lag %d' %lag)
        ix = inds[(inds+lag >= 0) & (inds+lag < NT)]  
        n_events = len(ix)   
        c, e = covariate[ix,...], events[ix+lag,...] # shapes (n_events, *any_shape), (n_events, <optional channels>)
        new_shapeC, new_shapeE = [n_events, 1, *sz], [n_events, NC, 1]
        et[:, i, ...] = reduction(c.reshape(new_shapeC) * e.reshape(new_shapeE), dim=0)
    if cov_ndims == 0:
        et = et.squeeze(-1)
    if events_ndims == 0:
        et = et.squeeze(0)
    return et

def model_device(model):
    return next(model.parameters()).device

def dl_device(dl):
    batch = next(iter(dl))
    if isinstance(batch, dict):
        return batch.values().__iter__().__next__().device
    return batch[0].device

def plot_transientsC_new(model, val_dl, cids, bins=(-40, 60), filter=True):
    assert issubclass(type(val_dl), DataLoader), "val_dl must be a DataLoader"
    assert issubclass(type(model), nn.Module), "model must be a nn.Module"
    assert len(cids) > 0, "cids must be a list of channel ids"
    assert len(bins) == 2, "bins must be a tuple of length 2"
    # assert model_device(model) == dl_device(val_dl), "Model and data must be on same device"
    assert hasattr(val_dl.dataset, 'covariates'), "val_dl.dataset must have covariates"
    device = model_device(model)
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
    transientY = event_triggered_op(Y, inds, bins, reduction=torch.mean).cpu()
    del Y
    transientY_hat = event_triggered_op(Y_hat, inds, bins, reduction=torch.mean).cpu()
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
        plt.title(cids[cc])
    plt.tight_layout()
    return transientY, transientY_hat, f
    

# def plot_transientsC(model, test_ds, cids, num_lags):
#     inds = torch.where(test_ds.covariates['sac_on'])[0].cpu().numpy()
#     N = len(inds)

#     Y = test_ds.covariates['robs']
#     bins = np.arange(-100, 150, 1)
#     Ndim = list(Y.shape[1:])
#     NT = Y.shape[0]

#     binnedY = np.nan*np.zeros([N] + [len(bins)] + Ndim)
#     good = np.zeros(N, dtype=bool)

#     from tqdm import tqdm
#     for i in tqdm(range(N)):
#         ii = inds[i] + bins
#         if np.min(ii) < 0 or np.max(ii) >= NT:
#             continue
        
#         good[i] = True
#         binnedY[i,...] = Y[ii,...].detach().cpu().numpy()

#     binnedY = binnedY[good,...]

#     plt.figure()
#     _ = plt.plot(bins, np.mean(binnedY, axis=0))
#     plt.xlabel("Time from saccade onset (ms)")


def plot_transientsC(model, val_dl, cids, num_lags):
    # val_dl must have batch size 1 and ds must have use_blocks = False
    Nfix = len(val_dl)
    n = []
    nbins = 120
    NC = len(cids)
    start = 0
    rsta = np.nan*np.ones( (Nfix, nbins, NC))
    dfs = np.nan*np.ones( (Nfix, nbins, NC))
    rhat = np.nan*np.ones( (Nfix, nbins, NC))
    esta = np.nan*np.ones( (Nfix, nbins, 2))
    for i,batch in enumerate(val_dl):
        batch = to_device(batch, next(model.parameters()).device)
        n_ = batch['eyepos'].shape[0]
        n.append(n_)
        nt = np.minimum(n_, nbins)
        yhat = model(batch)
        dfs[i,start:nt,:] = batch['dfs'][start:nt,cids].cpu()
        esta[i,start:nt,:] = batch['eyepos'][start:nt,:].cpu()
        rsta[i,start:nt,:] = batch['robs'][start:nt,cids].cpu()
        rhat[i,start:nt,:] = yhat[start:nt,:].detach().cpu().numpy()
        del batch
    
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    f = plt.figure(figsize=(10,10))
    for cc in range(NC):
        plt.subplot(sx,sy,cc+1)
        plt.plot(np.nansum(rsta[:,:,cc]*dfs[:,:,cc], axis=0)/np.nansum(dfs[:,:,cc]), 'k')
        plt.plot(np.nansum(rhat[:,:,cc]*dfs[:,:,cc], axis=0)/np.nansum(dfs[:,:,cc]), 'r')
        plt.xlim([num_lags, nbins])
        plt.axis('off')
        plt.title(cc)
    return rsta, rhat, f

def plot_transients_np(model, val_data, stimid=0, maxsamples=120):
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

def uneven_tqdm(iterable, total, get_len=lambda x: len(x["robs"]), **kwargs):
    """
        tqdm that works with uneven lengths of iterable and total
    """
    pbar = tqdm(total=total, **kwargs)
    for i in iterable:
        yield i
        pbar.update(get_len(i))
    pbar.close()
    
def reclass(obj, new_class_object=None):
    """
        Reclass an object to a new class
    """
    if new_class_object is None:
        try:
            new_class_object = globals()[obj.__class__.__name__]()
        except:
            print("if you are not providing an instance of the new class, you must ensure your new class is available in the global namespace.")
            print("Possibly cannot find class %s. make sure its imported directly." %obj.__class__.__name__)
            print("ensure that class __init__ works when not entering any arguments")
    for k, v in vars(obj).items():
        setattr(new_class_object, k, v)        
    return new_class_object

def show_stim_movie(stim, path="stim_video.mp4", fps=30, normalizing_constant=None):
    """
        Given stim show an interactive movie.
        
        stim: (time, 1, x, y)
    """
    stim = stim[:, 0].detach().cpu().numpy()
    if normalizing_constant is None:
        stim = stim/np.abs(stim).max()*127 + 127
    else:
        stim = stim * normalizing_constant + 127
    stim = stim.astype(np.uint8)
    writer = imageio.get_writer('test.mp4', fps=fps)
    for i in stim:
        writer.append_data(i)
    writer.close()
    w, h = stim.shape[1:]
    return Video("test.mp4", embed=True, width=w*3, height=h*3)

# Function to create a Gaussian kernel
def gaussian_kernel_1D(size: int, std: float):
    """Creates a 1D Gaussian kernel using the given size and standard deviation."""
    values = torch.arange(-size // 2 + 1., size // 2 + 1.)
    gauss_kernel = torch.exp(-values**2 / (2 * std**2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    return gauss_kernel

def gaussian_kernel_2D(size: int, std: float, channels: int):
    """Creates a 2D Gaussian kernel with the given size and standard deviation."""
    x = torch.linspace(-(size // 2), size // 2, steps=size)
    xx,yy = torch.meshgrid(x,x)
    gauss_kernel = torch.exp(-(xx**2 + yy**2) / (2 * std**2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    kernel = gauss_kernel.expand(1, channels, size, size)
    return kernel