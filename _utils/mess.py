from scipy.ndimage import gaussian_filter
import numpy as np
from NeuroVisKit._utils.utils import to_device
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb as wdb
import torch.nn as nn
import torch.nn.functional as F

def concat_dicts(*ds):
    return {k: torch.cat([d[k] for d in ds], dim=0) for k in ds[0].keys()}
            
class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

class Split(nn.Module):
    def __init__(self, interleave=True):
        super().__init__()
        self.interleave = interleave
    def forward(self, x):
        if self.interleave:
            return concatenate_interleave(x, -x, 1)
        
class Power(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, x):
        return x**self.p

class Padding(nn.Module):
    def __init__(self, k, op=nn.Identity()):
        super().__init__()
        self.k = k
        self.op = op
    def forward(self, x):
        return self.op(F.pad(x, (self.k, self.k, self.k, self.k)))

class Scaffold(nn.Module):
    def __init__(self, *ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
    def forward(self, x):
        outs = []
        for op in self.ops:
            x = op(x)
            outs.append(x)
        sz = min([x.shape[-1] for x in outs])
        outs = [F.interpolate(i, size=sz, mode='nearest') for i in outs]
        # outs.append(self.ops[-1](x))
        return torch.cat(outs, dim=1)

class HalfChannelOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
    def forward(self, x):
        c = x.shape[1] // 2
        return torch.cat([self.op(x[:, :c]), x[:, c:]], dim=1)

def fig_to_pil(fig):
    #matplotlib figure to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return wdb.Image(Image.open(buf))

def concatenate_interleave(t1, t2, dim):
    # Reshape t1 and t2 to have an additional dimension at the specified dimension
    t1 = t1.unsqueeze(dim + 1)
    t2 = t2.unsqueeze(dim + 1)
    
    # Concatenate along the new dimension and reshape the result
    interleaved = torch.cat((t1, t2), dim=dim + 1)
    interleaved = interleaved.reshape(list(t1.shape[:dim]) + [-1] + list(t1.shape[dim + 2:]))
    return interleaved

def concatenate_interleave_many(*ts, dim=0):
    # Reshape t1 and t2 to have an additional dimension at the specified dimension
    ts = [t.unsqueeze(dim + 1) for t in ts]
    interleaved = torch.cat(ts, dim=dim + 1)
    interleaved = interleaved.view(list(ts[0].shape[:dim]) + [-1] + list(ts[0].shape[dim + 2:]))
    return interleaved

def get_window(shape, dims=None, wfunc=torch.hamming_window):
    if dims is None:
        dims = [i for i in range(len(shape))]
    w = 1
    for i in dims:
        s = [1 for i in range(len(shape))]
        s[i] = -1
        w = w * wfunc(shape[i], periodic=False).reshape(s)
    return w

def get_conv_window(kshape, kdims=None, wfunc=torch.hamming_window):
    if kdims is None:
        kdims = [i for i in range(len(kshape))]
    kdims = [i+2 for i in kdims]
    return get_window([1, 1]+list(kshape), kdims, wfunc)

def window_tensor(x, dims=None, p_runtime=1, p_init=0):
    cls = type(x)
    def to(self, *args, **kwargs):
        a = cls.to(self, *args, **kwargs)
        a.w = a.w.to(*args, **kwargs)
        return a
    def windowed(self):
        return self.w**self.p_runtime * cls.data.__get__(self)
    setattr(x, "w", get_window(x.shape, dims=dims))
    setattr(x, "to", to)
    setattr(x, "windowed", property(fget=windowed))
    x.data = x.w**p_init * x.data
    return x

def get_gaus_kernel(shape):
    kernel = np.ones(shape)
    for i in range(len(shape)):
        # assert shape[i] % 2 == 1
        k = np.zeros(shape[i])
        k[shape[i]//2 + shape[i] % 2 - 1] = 1
        k = gaussian_filter(k, shape[i]/5).reshape(-1, *([1]*(len(shape)-1-i)))
        kernel = kernel * k
    return kernel
def index_into(obj, index, end=None):
    if isinstance(obj, dict):
        return {k: index_into(v, index, end=end) for k, v in obj.items()}
    if end is not None:
        return obj[index:end]
    return obj[index]
def len_(obj):
    if isinstance(obj, dict):
        return len(obj[list(obj.keys())[0]])
    return len(obj)
def split_batched_op(input, op, groups=2, device="cpu", inplace=False):
    b = len_(input)
    gsize = int(np.ceil(b / groups))
    if inplace:
        i = 0
        while i < b:
            iend = min(i+gsize, b)
            input[i:iend] = op(to_device(index_into(input, i, iend), device)).cpu()
            i = iend  
        return input  
    else:
        out = []
        i = 0
        while i < b:
            iend = min(i+gsize, b)
            out.append(op(to_device(index_into(input, i, iend), device)).cpu())
            i = iend
        return torch.cat(out, dim=0)

class IndexableDict(dict):
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return IndexableDict({k: v[key] for k, v in self.items()})
        return super().__getitem__(key)
def hann_window(shape):
    window = 1
    for i in range(len(shape)):
        window = window * torch.hann_window(shape[i]).reshape(-1, *([1]*(len(shape)-1-i)))
    return window
def interleave(a, b):
    # interleave two arrays across the first dimension
    return np.stack([a, b], axis=1).reshape(-1, *a.shape[1:])
def whiten(x, eps=1e-8):
    # use ZCA whitening on zscored data x shape (n, d)
    x = x - x.mean(0, keepdim=True)
    cov = x.T @ x / len(x)
    U, S, V = torch.svd(cov)
    return x @ U @ torch.diag(1/(S+eps).sqrt()) @ U.T
def angular_binning(x, bins=8, hrange=(-1, 1), wrange=(0, 1)):
    # x is (n, d)
    device = x.device if hasattr(x, "device") else "cpu"
    h = torch.linspace(hrange[1], hrange[0], x.shape[-2], device=device)
    w = torch.linspace(wrange[0], wrange[1], x.shape[-1], device=device)
    H, W = torch.meshgrid(h, w, indexing="ij")
    theta = torch.atan2(H, W)
    x = x**2
    x = x / x.sum()
    bins = np.linspace(-np.pi/2, np.pi/2, bins+1)
    hist, _ = np.histogram(theta.flatten(), bins, weights=x.flatten())
    return hist / hist.sum()
def angular_binning_kde(x, bins=8, hrange=(-1, 1), wrange=(0, 1)):
    # x is (n, d)
    device = x.device if hasattr(x, "device") else "cpu"
    from scipy.stats import gaussian_kde
    h = torch.linspace(hrange[1], hrange[0], x.shape[-2], device=device)
    w = torch.linspace(wrange[0], wrange[1], x.shape[-1], device=device)
    H, W = torch.meshgrid(h, w, indexing="ij")
    theta = torch.atan2(H, W)
    x = x**2
    if x.sum() == 0:
        return np.zeros(bins)
    if np.isnan(x.sum()):
        return np.zeros(bins) * np.nan
    x = x / x.sum()
    bins = np.linspace(-np.pi/2, np.pi/2, bins+1)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    if np.count_nonzero(x) == 1:
        theta_on = theta[np.argmax(x)]
        best_bin = np.argmin(np.abs(bins - np.array(theta_on)))
        out = np.zeros(len(bins))
        out[best_bin] = 1
        return out
    k = gaussian_kde(theta.flatten(), weights=x.flatten())
    hist = k(bins)
    return hist / hist.sum()
def radial_binning_kde(x, bins=8, hrange=(-1, 1), wrange=(0, 1)):
    # x is (n, d)
    device = x.device if hasattr(x, "device") else "cpu"
    from scipy.stats import gaussian_kde
    h = torch.linspace(hrange[1], hrange[0], x.shape[-2], device=device)
    w = torch.linspace(wrange[0], wrange[1], x.shape[-1], device=device)
    H, W = torch.meshgrid(h, w, indexing="ij")
    r = torch.sqrt(H**2 + W**2)
    x = x**2
    if x.sum() == 0:
        return np.zeros(bins)
    if np.isnan(x.sum()):
        return np.zeros(bins) * np.nan
    x = x / x.sum()
    bins = np.linspace(0, 1, bins+1)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    if np.count_nonzero(x) == 1:
        r_on = r[np.argmax(x)]
        best_bin = np.argmin(np.abs(bins - np.array(r_on)))
        out = np.zeros(len(bins))
        out[best_bin] = 1
        return out
    k = gaussian_kde(r.flatten(), weights=x.flatten())
    hist = k(bins)
    return hist / hist.sum()
def kde_ang(theta, robs, bins=8):
    # x is (n, d)
    if sum(robs) == 0:
        return np.zeros(bins)
    if not np.isfinite(robs.sum()):
        return np.zeros(bins) * np.nan
    from scipy.stats import gaussian_kde
    bins = np.linspace(0, 180, bins+1)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    if np.count_nonzero(robs) == 1:
        theta_on = theta[np.argmax(robs)]
        best_bin = np.argmin(np.abs(bins - np.array(theta_on)))
        out = np.zeros(len(bins))
        out[best_bin] = 1
        return out
    k = gaussian_kde(theta.flatten(), weights=robs.flatten())
    hist = k(bins)
    return hist / hist.sum()
def kde_rad(r, robs, bins=8):
    # x is (n, d)
    if sum(robs) == 0:
        return np.zeros(bins)
    if not np.isfinite(robs.sum()):
        return np.zeros(bins) * np.nan
    from scipy.stats import gaussian_kde
    bins = np.linspace(min(r), max(r), bins+1)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    if np.count_nonzero(robs) == 1:
        r_on = r[np.argmax(robs)]
        best_bin = np.argmin(np.abs(bins - np.array(r_on)))
        out = np.zeros(len(bins))
        out[best_bin] = 1
        return out
    k = gaussian_kde(r.flatten(), weights=robs.flatten())
    hist = k(bins)
    return hist / hist.sum()

def get_sta(stim, robs, modifier= lambda x: x, inds=None, lags=[8]):
    #added negative lag capability
    '''
    Compute the STA for a given stimulus and response
    stim: [N, C, H, W] tensor
    robs: [N, NC] tensor
    inds: indices to use for the analysis
    modifier: function to apply to the stimulus before computing the STA
    lags: list of lags to compute the STA over
    time_reversed: if True, compute the effect of robs on future stim

    returns: [NC, C, H, W, len(lags)] tensor
    '''

    if isinstance(lags, int):
        lags = [lags]
    
    if inds is None:
        inds = np.arange(stim.shape[0])

    NT = stim.shape[0]
    sz = list(stim.shape[1:])
    NC = robs.shape[1]
    sta = torch.zeros( [NC] + sz + [len(lags)], dtype=torch.float32)

    for i,lag in enumerate(lags):
        # print('Computing STA for lag %d' %lag)
        if lag >= 0:
            ix = inds[inds < NT-lag]
        else: 
            ix = inds[inds >= lag]
        sta[...,i] = torch.einsum('bchw, bn -> nchw', modifier(stim[ix,...]),  robs[ix+lag,:])/(NT-abs(lag))

    return sta