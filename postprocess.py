'''
    For now mainly focusing on integrated gradients.
'''
#%%
import torch
import numpy as np
import dill
import os
import math
from copy import deepcopy
from utils import unpickle_data, get_datasets, eval_model_summary, plot_layer
from models.utils.plotting import plot_sta_movie
from datasets.mitchell.pixel.utils import get_stim_list
from itertools import cycle
import matplotlib.pyplot as plt
%matplotlib inline
device = torch.device("cpu") #torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
nsamples_train=10000
nsamples_val=56643
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids = session['cids']
folderName = 'shifter_20200304_0'# 'goodmodel_20'
to_eval = True
#%%
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device, batch_size=1)

with open(os.path.join(dirname, folderName, 'model.pkl'), 'rb') as f:
    model = dill.load(f)
model.to(device)
if to_eval:
    eval_model_summary(model, val_dl)

#%%
core = model.model.core

#%%
loader = cycle(val_dl)
def nextX():
    return next(loader)
def nextXY():
    x = nextX()
    return x, x['robs'][..., cids]
def resetLoader():
    global loader
    loader = cycle(val_dl)

def getLayerGrad(layer_i):
    layer = core[layer_i]
    grads = layer.weight.grad
    if grads == None:
        return None
    num_filts = grads.shape[-1]
    return grads.detach().reshape(layer.filter_dims + [num_filts]).squeeze()

def forwardBackward(loss=True, neuron=None):
    model.zero_grad()
    if loss:
        X, Y = nextXY()
        Y_hat = model(X)
        loss = model.loss(Y_hat, Y, data_filters=X["dfs"][..., cids])
        loss.backward()
    else:
        y = model(nextX())
        if neuron == None:
            torch.mean(y).backward()
        else:
            y[0, neuron].backward()    

def gradSum(layer_i, nsamples=1000, loss=True):
    grads = torch.zeros(*core[layer_i].get_weights().shape, device=device)
    for i in range(nsamples):
        forwardBackward(loss)
        grads += torch.abs(getLayerGrad(layer_i))
    return grads / nsamples

def gradSum_all(nsamples=1000, loss=True):
    grads = [torch.zeros(*core[i].get_weights().shape, device=device) for i in range(len(core))]
    for i in range(nsamples):
        forwardBackward(loss)
        for layer_i in range(len(core)):
            grads[layer_i] += torch.abs(getLayerGrad(layer_i))
    grads = [i.detach().cpu().numpy()/nsamples for i in grads]
    return grads

def integratedGradients(n=25, neuron=None):
    model.zero_grad()
    X = nextX()
    stim0 = deepcopy(X['stim'])
    for i in range(n):
        # x = deepcopy(X)
        X["stim"] = stim0 * (i / n)
        y = model(X)
    if neuron == None:
        torch.mean(y).backward()
    else:
        y[0, neuron].backward()  
    grads = [getLayerGrad(i) for i in range(len(core))] 
    return [i.detach().cpu().numpy()/n for i in grads]

def getLayerDist(layer):
    units = np.abs(layer).mean((0, 1, 2))
    quantiles = np.percentile(units, [0, 25, 50, 75, 100])
    print(quantiles)
    quantiles[:2] = quantiles[2]-quantiles[:2]
    quantiles[3:] = quantiles[3:]-quantiles[2]
    return quantiles, units.mean()

# def plotGradDist(grads):
#     plt.figure()
#     plt.title('Gradient Distribution (magnitude)')
#     plt.xlabel('Layer')
#     plt.ylabel('Relative gradient mag')
#     # out = [getLayerDist(layer) for layer in grads]
#     # out_max = max([i.max() for i in grads])
#     # out = [[d/out_max for d in i] for i in out]
#     # minmax_error = [[i[0][0] for i in out], [i[0][4] for i in out]]
#     # quantile_error = [[i[0][1] for i in out], [i[0][3] for i in out]]
#     # categories = [str(i) for i in range(len(out))]
#     # plt.bar(categories, [i[0][2] for i in out], yerr=minmax_error, capsize=5)
#     # plt.bar(categories, [i[0][2] for i in out], yerr=quantile_error, capsize=5, ecolor='r')
#     # plt.scatter(range(len(out)), [i[1] for i in out], c='blue', marker='x')
#     plt.boxplot(grads)

def plotGrads(grads):
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
        x = np.random.normal(i+1, 0.05, len(gradi))
        plt.scatter(x, gradi, marker='x')

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
        plt.subplot(2, 1, 2)
        plt.title('per-layer normalization')
        plt.ylabel('Layer')
        plt.xlabel('Relative mag')
        plt.imshow(grads, cmap='hot', interpolation='None', origin='lower')
        plt.colorbar()

    plt.tight_layout()

def getIntegratedGradSummary(nsamples=1000):
    grads = integratedGradients()
    for i in range(1, nsamples):
        new_grads = integratedGradients()
        grads = [grads[i] + new_grads[i] for i in range(len(grads))]
    grads = [i/nsamples for i in grads]
    plotGrads(grads)
    return grads

def getGradSummary(loss=True, sample_points=[1, 100, 1000, 10000], stability=False):
    plt.figure()
    plt.suptitle('Gradient Sparsity (entire units)')
    n = len(sample_points)
    sx = math.floor(np.sqrt(n))
    sy = math.ceil(n / sx)
    stability = []
    plota = []
    for point in sample_points:
        resetLoader()
        grads = gradSum_all(nsamples=point, loss=loss)
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


#%%
plot_layer(core[0])

w = np.transpose(core[0].get_weights(), (2,0,1,3))
w = np.concatenate((np.zeros( [1] + list(w.shape[1:])), w))
plot_sta_movie(w, frameDelay=2, path=folderName+'.gif')

#%%
grads = getIntegratedGradSummary(nsamples=100)

#%%

getGradSummary(True)
getGradSummary(False)
# %%
from models.utils import plot_stas
plot_stas(np.transpose(grads[0], (2, 0, 1, 3)))

#%%
plot_stas(grads[1])
# %%

def plotTransients(stimid=0, maxsamples=120):
    from tqdm import tqdm
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

    plt.figure(figsize=(sx*2, sy*2))
    for cc in range(NC):
        
        plt.subplot(sx, sy, cc + 1)
        _ = plt.plot(np.nanmean(sta_true[:,:,cc],axis=0), 'k')
        _ = plt.plot(np.nanmean(sta_hat[:,:,cc],axis=0), 'r')
        plt.axis("off")
        plt.title(cc)

    plt.show()

    return sta_true, sta_hat

sta_true, sta_hat = plotTransients()

#%%


cc = 37
i += 1
plt.plot(sta_true[i,:,cc], 'k')
plt.plot(sta_hat[i,:,cc], 'r')


    



# %%
