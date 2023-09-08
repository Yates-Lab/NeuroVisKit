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
from .mei import irf, get_gratings, irfC
from .cluster import corr_dist, cos_dist
from .utils import to_device
from scipy.signal import find_peaks
import torch.nn as nn

def plot_model_conv(model):
    i = 0
    for module in model.modules():
        # check if convolutional layer
        if issubclass(type(module), nn.modules.conv._ConvNd):
            w = module.weight.data.cpu().numpy()
            if len(w.shape) == 5: # 3d conv (cout, cin, x, y, t)
                w = w.squeeze(1) # asume cin is 1
                w = w/np.abs(w).max((1, 2, 3), keepdims=True) # normalize xyt
            elif len(w.shape) == 4: # 2d conv (cout, cin, x, y)
                w = w/np.abs(w).max((1, 2, 3), keepdims=True) # normalize cin xy
            # shape is (cout, cin, x, y)
            titles = ['cout %d'%i for i in range(w.shape[0])]
            plot_grid(w, titles=titles, suptitle='Layer %d'%i, desc='Layer %d'%i, vmin=-1, vmax=1)
            i += 1
            
class Loader():
    def __init__(self, ds, cyclic=True, shuffled=False):
        self.ds = ds
        self.inds = np.arange(len(ds))
        if shuffled:
            np.random.shuffle(self.inds)
        self.iter = cycle if cyclic else iter
        self.loader = self.iter(self.inds)
    def __next__(self):
        return self.ds[next(self.loader)]
    def reset(self):
        self.loader = self.iter(self.inds)

class Grad():
    def __init__(self, grads):
        self.grads = grads
    def __getitem__(self, key):
        return self.grads[key]
    def abs(self):
        return Grad([g.abs() for g in self.grads])    
    def __div__(self, other):
        return Grad([g/other for g in self.grads])
    def __mul__(self, other):
        return Grad([g*other for g in self.grads])
    def __add__(self, other):
        return Grad([g+other for g in self.grads])
    def __sub__(self, other):
        return Grad([g-other for g in self.grads])

def plot_grid(mat, titles=None, vmin=None, vmax=None, desc='Grid plot', **kwargs):
    '''
        Plot a grid of figures such that each subfigure has m subplots.
        mat is a list of lists of image data (n, m, x, y)
        titles is a list of titles of length n.
    '''
    n = len(mat)
    m = len(mat[0])
    
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(2*m, 2*n))
    
    for i in tqdm(range(n), desc=desc):
        for j in range(m):
            axes[i, j].imshow(mat[i][j], vmin=vmin, vmax=vmax, interpolation='none')
            axes[i, j].axis('off')

            if titles is not None:
                axes[i, j].set_title(titles[i])
    
    for key in kwargs:
        eval(f'plt.{key}')(kwargs[key])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
    
def eval_gratings(model, device='cpu', alpha=1, shape=(35, 35, 24), fs=None):
    '''
    Evaluate the model on cosine gratings.
    alpha: scaling factor for the gratings (0, 1]
    fs: frequency samples for each dimension
    '''
    if fs is None:
        fs = shape
    fx = np.linspace(0, 0.5, fs[0])
    fy = np.linspace(0, 0.5, fs[1])
    ft = np.linspace(0, 0.5, fs[2])
    frequencies = np.stack(np.meshgrid(fx, fy, ft), -1).reshape(-1, 3)
    evals, phases = [], []
    for ind, freq_ind in enumerate(frequencies):
        freqs, gratings = get_gratings(shape, *np.array(freq_ind).reshape(3, 1))
        gratings = torch.tensor(gratings, device=device).reshape(-1, *shape)
        model.to(device)
        out = model({"stim": gratings*alpha})
        mx, amx = out.max(0)
        phases.append(freqs[amx.cpu(), 3:])
        evals.append(mx.detach().cpu())
        frequencies[ind] = freqs[0, :3]
        del out, freqs, gratings, mx, amx
        print(f'{ind+1} / {frequencies.shape[0]}')
        
    return frequencies.reshape(*fs, 3), np.stack(phases, 1).reshape(-1, *fs, 3), torch.stack(evals).reshape(*fs, -1).detach().numpy()

def csf_kernel_from_activations(phases, activations, shape=(35, 35, 24), fs=None):
    if fs is None:
        fs = shape
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    t = np.arange(shape[2])
    fx = np.linspace(0, 0.5, fs[0])
    fy = np.linspace(0, 0.5, fs[1])
    ft = np.linspace(0, 0.5, fs[2])
    X, Y, T = np.stack(np.meshgrid(x, y, t)).reshape(3, *shape, 1)
    frequencies = np.stack(np.meshgrid(fx, fy, ft), -1).reshape(-1, 3)
    phases = phases.reshape(phases.shape[1], -1, 3)
    out_kernel = np.zeros((shape[0], shape[1], shape[2], activations.shape[-1]))  
    activations = activations.reshape(-1, activations.shape[-1]) 
    for ind, freq_ind in enumerate(frequencies):
        FX, FY, FT = freq_ind
        PHIx, PHIy, PHIt = phases[:, ind].T.reshape(3, 1, 1, 1, -1)
        new_kernel = np.cos(2*np.pi*FX*X+PHIx)*np.cos(2*np.pi*FY*Y+PHIy)*np.cos(2*np.pi*FT*T+PHIt)
        out_kernel += new_kernel*activations[ind]
    return out_kernel.reshape(*shape, -1)


def plot_contour_peaks(axs, n=1):
    contours = [i for i in axs.get_children() if type(i) == matplotlib.collections.PathCollection]
    levels = [[path._vertices.mean(0).tolist() for path in cont._paths] for cont in contours][::-1]
    lengths = [len(i) for i in levels]
    levels_n = [levels[i] for i in range(1, len(levels)) if lengths[i]>max(lengths[:i])]
    zero = np.array(levels[0]).reshape(-1, 2)
    for i in range(len(levels_n)):
        current = levels_n[i]
        dists = [min(np.linalg.norm(zero-j, axis=1)) for j in current]
        inds = np.argsort(dists)[::-1][:len(current)-len(levels[0])]
        levels_n[i] = [current[ind] for ind in inds]
    levels = [levels[0]] + [i for i in levels_n if len(i)>0]
    levels = np.concatenate(levels[:min(n, len(levels))], 0)
    axs.scatter(*levels.T, c='black')
    return levels

def hist_peaks(distances, to_plot=False, **kwargs):
    if to_plot:
        plt.figure()
        counts, bins, _ = plt.hist(distances, **kwargs)
    else:
        counts, bins = np.histogram(distances, **kwargs)
    mid_bins = (bins[1:] + bins[:-1])/2
    peaks, _ = find_peaks(counts, height=counts.max()/8)
    return peaks, mid_bins[peaks], counts[peaks]

# def eucl(x, y):
#     out = x.dot(y)
#     norm = (x.dot(x) * y.dot(y))**0.5
#     return out/norm/2 + 0.5

def get_peak_irfs(irfs):
    dists_eucl, dists_corr = [], []
    for i in irfs[1:]:
        dists_eucl.append(cos_dist(i, irfs[0]))
        dists_corr.append(corr_dist(i, irfs[0]))
    dists_eucl = np.array(dists_eucl)
    dists_corr = np.array(dists_corr)
    eucl_hist_peaks = hist_peaks(dists_eucl, bins=20)
    corr_hist_peaks = hist_peaks(dists_corr, bins=20)
    eucl_irfs, corr_irfs = [], []
    for i in eucl_hist_peaks[1]:
        eucl_irfs.append(irfs[np.argmin(np.abs(dists_eucl - i))])
    for i in corr_hist_peaks[1]:
        corr_irfs.append(irfs[np.argmin(np.abs(dists_corr - i))])
    states = f'{eucl_hist_peaks[0].shape[0]} eucl states, {corr_hist_peaks[0].shape[0]} corr states'
    return {
        'peaks': (eucl_hist_peaks, corr_hist_peaks),
        'irfs': (eucl_irfs, corr_irfs),
        'zero': irfs[0],
        'num_states': (eucl_hist_peaks[0].shape[0], corr_hist_peaks[0].shape[0]),
        'string': states
    }
    
def get_zero_irf(input_dims, model, cid, device='cpu', nobatch=False):
    dims = input_dims if nobatch else (1, *input_dims)
    return irf(
        {
            "stim":
                torch.zeros(dims, device=device)
        }, model, [cid]).detach().cpu().squeeze(0)
    
def get_zero_irfC(input_dims, model, cid, device='cpu'):
    rf = irfC(
        {
            "stim":
                torch.zeros(input_dims, device=device)
        }, model, [cid], input_dims[0])[0]
    return rf.detach().cpu()

def generate_irfs(val_dl, model, cids, path=None, zero=False, device='cpu'):
    '''
        Generates IRFs for a given neuron id, model and dataset
        saves the IRFs to path if provided
        Includes the IRF for when input is all zeros if zero=True
    '''
    irfs = []
    for batch in tqdm(val_dl):
        irfs.append(irf(batch, model, cids).detach().cpu())
    if zero:
        irfs.insert(0, irf(
            {
                "stim":
                    torch.zeros((1, irfs[0].shape[-1]), device=device)
            }, model, cids).detach().cpu())
    irfs = torch.cat(irfs, dim=0)
    if path is not None:
        torch.save(irfs, path)
    return irfs
    
def zscoreWeights(w):
    w_mean = np.mean(w, axis=(0, 1, 2), keepdims=True)
    w_std = np.std(w, axis=(0, 1, 2), keepdims=True)
    w_normed = (w - w_mean) / w_std
    return w_normed

def eval_model_dist(dirname, nsamples_train=None, nsamples_val=None, device=torch.device('cpu')):
    train_data, val_data = utils.unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
    train_dl, val_dl, train_ds, val_ds = utils.get_datasets(train_data, val_data, device=device, batch_size=1)

    evals = []
    for i in range(100):
        print(f'Started {i}')
        folderName = f'checkpoint_{i}'# 'goodmodel_20'
        with open(os.path.join(dirname, folderName, 'model.pkl'), 'rb') as f:
            model = dill.load(f)
        model.to(device)
        evals.append(eval_model_fast(model, val_dl))
        print(evals[-1])

    dill.dump(evals, open(os.path.join(dirname, 'evals.pkl'), 'wb'))

def plot_model_dist(dirname):
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

def next_XY(loader, cids):
    x = next(loader)
    return x, x['robs'][..., cids]

def get_layer_grad(core, layer_i):
    layer = core[layer_i]
    grads = layer.weight.grad
    if grads == None:
        return None
    num_filts = grads.shape[-1]
    return grads.detach().reshape(layer.filter_dims + [num_filts]).squeeze()

def forward_back(loader, model, cids, loss=True, neuron=None):
    model.zero_grad()
    if loss:
        X, Y = next_XY(loader, cids)
        Y_hat = model(X)
        loss = model.loss(Y_hat, Y, data_filters=X["dfs"][..., cids])
        loss.backward()
    else:
        y = model(next(loader))
        if neuron == None:
            torch.mean(y).backward()
        else:
            y[0, neuron].backward()    

def grad_sum_layer(model, core, loader, layer_i, device, cids, nsamples=1000, loss=True):
    grads = torch.zeros(*core[layer_i].get_weights().shape, device=device)
    for i in range(nsamples):
        forward_back(loader, model, cids, loss)
        grads += torch.abs(get_layer_grad(core, layer_i))
    return grads / nsamples

def grad_sum(model, core, loader, device, cids, nsamples=1000, loss=True):
    grads = [torch.zeros(*core[i].get_weights().shape, device=device) for i in range(len(core))]
    for i in range(nsamples):
        forward_back(loader, model, cids, loss)
        for layer_i in range(len(core)):
            grads[layer_i] += torch.abs(get_layer_grad(core, layer_i))
    grads = [i.detach().cpu().numpy()/nsamples for i in grads]
    return grads

def integrated_gradients(model, core, loader, n=25, neuron=None):
    model.zero_grad()
    X = next(loader)
    stim0 = deepcopy(X['stim'])
    for i in range(n):
        # x = deepcopy(X)
        X["stim"] = stim0 * (i / n)
        y = model(X)
    if neuron == None:
        torch.mean(y).backward()
    else:
        y[0, neuron].backward()  
    grads = [get_layer_grad(core, i) for i in range(len(core))] 
    return [i.detach().cpu().numpy()/n for i in grads]

def get_layer_dist(layer):
    units = np.abs(layer).mean((0, 1, 2))
    quantiles = np.percentile(units, [0, 25, 50, 75, 100])
    print(quantiles)
    quantiles[:2] = quantiles[2]-quantiles[:2]
    quantiles[3:] = quantiles[3:]-quantiles[2]
    return quantiles, units.mean()

def plot_grads(grads):
    rowNum = 2 if all([len(i) == len(grads[0]) for i in grads]) else 1
    plt.figure(figsize=(10, 5*rowNum))
    plt.suptitle("Gradient Distribution (magnitude)")
    grads = [np.abs(i).mean((0, 1, 2)) for i in grads]
    plt.subplot(rowNum, 2, 1)
    plt.title('Not normalized')
    plt.xlabel('Layer')
    plt.ylabel('mag')
    plt.boxplot(grads)
    for i, gradi in enumerate(grads):
        x = np.random.normal(i, 0.05, len(gradi))
        plt.scatter(x, gradi, marker='x')

    grads_sorted = [np.sort(i, axis=0) for i in grads]
    grads = [i / i.max() for i in grads]
    plt.subplot(rowNum, 2, 2)
    plt.title('per-layer normalization')
    plt.xlabel('Layer')
    plt.ylabel('Relative mag')
    plt.boxplot(grads)
    for i, gradi in enumerate(grads):
        x = np.random.normal(i+1, 0.05, len(gradi))
        plt.scatter(x, gradi, marker='x')

    if rowNum == 2:
        plt.subplot(4, 1, 3)
        plt.title('per-layer normalization')
        plt.ylabel('Layer')
        plt.gca().set_xticks([])
        grads_im = [np.sort(i, axis=0) for i in grads]
        plt.imshow(grads_im, cmap='hot', interpolation='None', origin='lower')
        plt.colorbar()

        plt.subplot(4, 1, 4)
        plt.title('global normalization')
        plt.ylabel('Layer')
        plt.gca().set_xticks([])
        mx = max([i.max() for i in grads_sorted])
        plt.imshow([i/mx for i in grads_sorted], cmap='hot', interpolation='None', origin='lower')
        plt.colorbar()

    plt.tight_layout()

def get_integrated_grad_summary(model, core, loader, nsamples=1000):
    grads = integrated_gradients(model, core, loader)
    for i in range(1, nsamples):
        new_grads = integrated_gradients(model, core, loader)
        grads = [grads[i] + new_grads[i] for i in range(len(grads))]
    grads = [i/nsamples for i in grads]
    plot_grads(grads)
    return grads

def get_grad_summary(model, core, loader, device, cids, loss=True, sample_points=[1, 100, 1000, 10000], stability=False):
    plt.figure()
    plt.suptitle('Gradient Sparsity (entire units)')
    n = len(sample_points)
    sx = math.floor(np.sqrt(n))
    sy = math.ceil(n / sx)
    stability = []
    plota = []
    for point in sample_points:
        loader.reset()
        grads = grad_sum(model, core, loader, device, cids, nsamples=point, loss=loss)
        s = [np.abs(layer).mean((0, 1, 2)) for layer in grads]
        stability.append([(layer!=0).astype(int) for layer in s])
        nz = [np.count_nonzero(layer)/len(layer) for layer in s]
        plota.append(nz)
        plt.subplot(sx, sy, sample_points.index(point) + 1)
        plt.stem(nz)
        plt.title(f'{point} samples')
        plt.xlabel('layer')
        plt.ylabel('frac nz grads')
        plt.ylim(0, 1)
    plt.tight_layout()

    # #TODO Check this
    # if stability:
    #     stab_per_layer = [[] for i in range(len(stability[0]))]
    #     for i in range(1, len(stability)):
    #         diff = [stability[i][layer] - stability[i-1][layer] for layer in range(len(stability[i]))]
    #         for layer in range(len(diff)):
    #             stab_per_layer[layer].append(np.abs(diff[layer]).mean())
    #     plt.figure()
    #     plt.suptitle('Gradient Stability (entire units)')
    #     for i, layer in enumerate(stab_per_layer):
    #         plt.subplot(sx, sy, i + 1)
    #         plt.stem(layer)
    #         plt.title(f'layer {i}')
    #         plt.xlabel('sample points (log scale)')
    #         plt.ylabel('frac changed grads')
    #         plt.ylim(0, 1)
    #     plt.tight_layout()

def plot_layer(layer, ind=None):
    
    if layer.filter_dims[-1] > 1:
        ws = layer.get_weights()
        if ind is not None: 
            ws = ws[..., ind]
        ws = np.transpose(ws, (2,0,1,3))
        layer.plot_filters()
    else:
        ws = layer.get_weights()
        if ind is not None: 
            ws = ws[..., ind]

        nout = ws.shape[-1]
        nin = ws.shape[0]
        plt.figure(figsize=(10, 10))

        for i in range(nin):
            for j in range(nout):
                plt.subplot(nin, nout, i*nout + j + 1)
                plt.imshow(ws[i, :,:, j], aspect='auto', interpolation='none')
                plt.axis("off")

def plot_dense_readout(layer):
    ws = layer.get_weights()
    n = ws.shape[-1]
    sx = int(np.ceil(np.sqrt(n)))
    sy = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(sx*2, sy*2))
    for cc in range(n):
        plt.subplot(sx, sy, cc + 1)
        v = np.max(np.abs(ws[:,:,cc]))
        plt.imshow(ws[:,:,cc], interpolation='none', cmap=plt.cm.coolwarm, vmin=-v, vmax=v)

def plot_model(model):

    for layer in model.core:
        plot_layer(layer)
        plt.show()

    if hasattr(model, 'offsets'):
        for i,layer in enumerate(model.offsets):
            _ = plt.plot(layer.get_weights())
            plt.title("Offset {}".format(model.offsetstims[i]))
            plt.show()
    
    if hasattr(model, 'gains'):
        for i,layer in enumerate(model.gains):
            _ = plt.plot(layer.get_weights())
            plt.title("Gain {}".format(model.gainstims[i]))
            plt.show()

    if hasattr(model.readout, 'mu'):
        plt.imshow(model.readout.get_weights())
    elif hasattr(model.readout, 'space'):
        plot_dense_readout(model.readout.space)
        plt.show()
        plt.imshow(model.readout.feature.get_weights())
        plt.xlabel("Neuron ID")
        plt.ylabel("Feature ID")
    
def eval_model(model, valid_dl):
    loss = model.loss.unit_loss
    model.eval()

    LLsum, Tsum, Rsum = 0, 0, 0
        
    device = next(model.parameters()).device  # device the model is on
    with torch.no_grad():
        if isinstance(valid_dl, dict):
            for dsub in valid_dl.keys():
                    if valid_dl[dsub].device != device:
                        valid_dl[dsub] = valid_dl[dsub].to(device)
            rpred = model(valid_dl)
            LLsum = loss(rpred,
                        valid_dl['robs'][:,model.cids],
                        data_filters=valid_dl['dfs'][:,model.cids],
                        temporal_normalize=False)
            Tsum = valid_dl['dfs'][:,model.cids].sum(dim=0)
            Rsum = (valid_dl['dfs'][:,model.cids]*valid_dl['robs'][:,model.cids]).sum(dim=0)

        else:
            for data in tqdm(valid_dl, desc='Eval models'):
                        
                for dsub in data.keys():
                    if data[dsub].device != device:
                        data[dsub] = data[dsub].to(device)
                
                rpred = model(data)
                LLsum += loss(rpred,
                        data['robs'][:,model.cids],
                        data_filters=data['dfs'][:,model.cids],
                        temporal_normalize=False)
                Tsum += data['dfs'][:,model.cids].sum(dim=0)
                Rsum += (data['dfs'][:,model.cids] * data['robs'][:,model.cids]).sum(dim=0)
                
    LLneuron = LLsum/Rsum.clamp(1)

    rbar = Rsum/Tsum.clamp(1)
    LLnulls = torch.log(rbar)-1
    LLneuron = -LLneuron - LLnulls

    LLneuron/=np.log(2)

    return LLneuron.detach().cpu().numpy()

def eval_model_fast(model, valid_data, t_mean = 0, t_std = 1):
    '''
        Evaluate model on validation data.
        valid_data: either a dataloader or a dictionary of tensors.
    '''
    loss = model.loss.unit_loss
    model.eval()
    LLsum, Tsum, Rsum = 0, 0, 0
    if isinstance(valid_data, dict):
        with torch.no_grad():
            rpred = model(valid_data) * t_std + t_mean
            LLsum = loss(rpred,
                        valid_data['robs'][:,model.cids],
                        data_filters=valid_data['dfs'][:,model.cids],
                        temporal_normalize=False)
            Tsum = valid_data['dfs'][:,model.cids].sum(dim=0)
            Rsum = (valid_data['dfs'][:,model.cids]*valid_data['robs'][:,model.cids]).sum(dim=0)
    else:
        for data in tqdm(valid_data, desc='Eval models'):  
            data = to_device(data, next(model.parameters()).device)          
            with torch.no_grad():
                rpred = model(data) * t_std + t_mean
                LLsum += loss(rpred,
                        data['robs'][:,model.cids],
                        data_filters=data['dfs'][:,model.cids],
                        temporal_normalize=False)
                Tsum += data['dfs'][:,model.cids].sum(dim=0)
                Rsum += (data['dfs'][:,model.cids] * data['robs'][:,model.cids]).sum(dim=0)
            del data
                
    LLneuron = LLsum/Rsum.clamp(1)

    rbar = Rsum/Tsum.clamp(1)
    LLnulls = torch.log(rbar)-1
    LLneuron = -LLneuron - LLnulls

    LLneuron/=np.log(2)

    return LLneuron.detach().cpu().numpy()

def eval_model_summary(model, valid_dl, **kwargs):
    ev = eval_model_fast(model, valid_dl, **kwargs)
    print(ev)
    if np.inf in ev or np.nan in ev:
        i = np.count_nonzero(np.isposinf(ev))
        ni = np.count_nonzero(np.isneginf(ev))
        print(f'Warning: {i} neurons have infinite/nan bits/spike, and {ni} neurons have ninf.')
        ev = ev[~np.isinf(ev) & ~np.isnan(ev)]
    # Creating histogram
    _, ax = plt.subplots()
    ax.hist(ev, bins=10)
    plt.axvline(x=np.max(ev), color='r', linestyle='--')
    plt.axvline(x=np.min(ev), color='r', linestyle='--')
    plt.xlabel("Bits/spike")
    plt.ylabel("Neuron count")
    plt.title("Model performance")
    # Show plot
    plt.show()
    return ev
# %%
