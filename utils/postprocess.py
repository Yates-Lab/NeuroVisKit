import dill
import os
import torch
import numpy as np
import math
from copy import deepcopy
from itertools import cycle
import matplotlib.pyplot as plt
from . import utils

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
        evals.append(eval_model(model, val_dl))
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

class Loader():
    def __init__(self, dl, cycle=True):
        self.dl = dl
        self.loader = None
        self.cycle = cycle
        self.reset()
    def __next__(self):
        return next(self.loader)
    def reset(self):
        self.loader = cycle(self.dl) if cycle else iter(self.dl)

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

    grads_sorted = np.sort(np.array(grads), axis=1)
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
        grads_im = np.sort(np.array(grads), axis=1)
        plt.imshow(grads_im, cmap='hot', interpolation='None', origin='lower')
        plt.colorbar()

        plt.subplot(4, 1, 4)
        plt.title('global normalization')
        plt.ylabel('Layer')
        plt.gca().set_xticks([])
        plt.imshow(grads_sorted/grads_sorted.max(), cmap='hot', interpolation='None', origin='lower')
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

    #TODO Check this
    if stability:
        stab_per_layer = [[] for i in range(len(stability[0]))]
        for i in range(1, len(stability)):
            diff = [stability[i][layer] - stability[i-1][layer] for layer in range(len(stability[i]))]
            for layer in range(len(diff)):
                stab_per_layer[layer].append(np.abs(diff[layer]).mean())
        plt.figure()
        plt.suptitle('Gradient Stability (entire units)')
        for i, layer in enumerate(stab_per_layer):
            plt.subplot(sx, sy, i + 1)
            plt.stem(layer)
            plt.title(f'layer {i}')
            plt.xlabel('sample points (log scale)')
            plt.ylabel('frac changed grads')
            plt.ylim(0, 1)
        plt.tight_layout()

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
    else:
        plot_dense_readout(model.readout.space)
        plt.show()
        plt.imshow(model.readout.feature.get_weights())
        plt.xlabel("Neuron ID")
        plt.ylabel("Feature ID")
    
def eval_model(model, valid_dl):
    loss = model.loss.unit_loss
    model.eval()

    LLsum, Tsum, Rsum = 0, 0, 0
    from tqdm import tqdm
        
    device = next(model.parameters()).device  # device the model is on
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
            
            with torch.no_grad():
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

def eval_model_summary(model, valid_dl):
    ev = eval_model(model, valid_dl)
    if np.inf in ev:
        i = np.count_nonzero(np.isposinf(ev))
        ni = np.count_nonzero(np.isneginf(ev))
        print(f'Warning: {i} neurons have infinite bits/spike, and {ni} neurons have ninf.')
        ev = ev[~np.isinf(ev)]
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