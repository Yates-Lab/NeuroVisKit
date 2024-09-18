from scipy.ndimage import gaussian_filter
import numpy as np
from NeuroVisKit._utils.utils import to_device
import torch

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