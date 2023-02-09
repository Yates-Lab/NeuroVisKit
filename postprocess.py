'''
    Analysis of trained models.
'''
#TODO Make it spit out a PDF and also make it compatible with models that have nonuniform layers
#%%
import torch
import numpy as np
import dill
import os
from utils.utils import unpickle_data, get_datasets, plot_transients
from models.utils import plot_stas
import utils.postprocess as utils
from models.utils.plotting import plot_sta_movie
import matplotlib.pyplot as plt
# %matplotlib inline

device = torch.device("cpu") #torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
nsamples_train=10000
nsamples_val=56643
RUN_NAME = 'test'
to_eval = [False, True, 'distribution'][1]

cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids = session['cids']
# RUN_NAME = 'shifter_20200304_0_window_valid'# 'goodmodel_20'
if to_eval == 'distribution':
    utils.eval_model_dist(dirname)
    utils.plot_model_dist(dirname)
    quit()
#%%
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device, batch_size=1)
loader = utils.Loader(val_dl)
#%%
with open(os.path.join(dirname, RUN_NAME, 'model.pkl'), 'rb') as f:
    model = dill.load(f)
model.to(device)
if to_eval:
    utils.eval_model_summary(model, val_dl)

#%%
core = model.model.core

#%%
utils.plot_layer(core[0])
#%%
w = core[0].get_weights()
# w = (w - np.mean(w, axis=(0, 1, 2))) / np.std(w, axis=(0, 1, 2))
# w = np.transpose(core[0].get_weights(), (2,0,1,3))
# w = np.concatenate((np.zeros( [1] + list(w.shape[1:])), w))
savePath = os.path.join(cwd, 'data', RUN_NAME)
plot_sta_movie(w, frameDelay=1, path=savePath+'2D.gif')
plot_sta_movie(w, frameDelay=1, path=savePath+'3D.gif', threeD=True)

#%%
grads = utils.get_integrated_grad_summary(model, core, loader, nsamples=100)
utils.get_grad_summary(model, core, loader, device, cids, True)
utils.get_grad_summary(model, core, loader, device, cids, False)
# %%
_ = plot_stas(np.transpose(grads[0], (2, 0, 1, 3)))
_ = plot_stas(grads[1])
# %%

sta_true, sta_hat, _ = plot_transients(model, val_data)

#%%
i = 0
#%%

cc = 37
i += 1
plt.plot(sta_true[i,:,cc], 'k')
plt.plot(sta_hat[i,:,cc], 'r')

#%%
utils.plot_model(model.model)
# %%
