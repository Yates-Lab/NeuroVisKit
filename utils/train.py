import torch
import dill
import NDNT
from utils.trainer import train
from models import ModelWrapper, CNNdense
from utils.get_models import get_cnn
from utils.utils import seed_everything
from dadaptation import DAdaptAdam, DAdaptSGD, DAdaptAdaGrad
from dog import DoG, LDoG
import torch.nn as nn
import torch.nn.functional as F

def smooth_robs(x, smoothN=10):
    smoothkernel = torch.ones((1, 1, smoothN, 1), device=device) / smoothN
    out = F.conv2d(
        F.pad(x, (0, 0, smoothN-1, 0)).unsqueeze(0).unsqueeze(0),
        smoothkernel).squeeze(0).squeeze(0)  
    assert len(x) == len(out)
    return out

def zscore_robs(x):
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True)

def train_f(opt, **kwargs):
    def train_f(model, train_loader, val_loader, checkpoint_dir, device, patience=30):
        max_epochs = 100
        optimizer = opt(model.parameters(), **kwargs)
        val_loss_min = train(
            model.to(device),
            train_loader,
            val_loader,
            optimizer=optimizer,
            max_epochs=max_epochs,
            verbose=2,
            checkpoint_path=checkpoint_dir,
            device=device,
            patience=patience)
        return val_loss_min
    return train_f

TRAINER_DICT = {
    'adam': train_f(torch.optim.Adam, lr=0.001),
    # 'lbfgs': train_lbfgs,
    'dadaptadam': train_f(DAdaptAdam, lr=1),
    'dadaptsgd': train_f(DAdaptSGD, lr=1),
    'dadaptadagrad': train_f(DAdaptAdaGrad, lr=1),
    'dog': train_f(DoG, lr=1),
    'ldog': train_f(LDoG, lr=1),
}

def get_trainer(config):
    return TRAINER_DICT[config['trainer']]

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# robs = train_data["robs"].clone()[30000:40000]
# mx = [robs[:, i][robs[:, i]>1].sum()/len(robs) for i in range(robs.shape[1])]
# print(max(mx), "max portion of nonzero spikes")
# print(np.mean(mx), "avg portion of nonzero spikes")
# print(np.std(mx), "std portion of nonzero spikes")
# print(np.median(mx), "med portion of nonzero spikes")

# # fsize = 1
# # for i in range(len(robs)-1, fsize-1, -1):
# #     robs[i] = robs[i-fsize:i].mean(0)
# # # for i in [30, 37, 22, 26, 41, 10][:1]:
# # i=22
# # c = 1000
# # plt.figure(figsize=(10, 2*len(robs)//c))
# # for j in range(0, len(robs), c):
# #     plt.subplot(len(robs)//c, 1, j//c+1)
# #     robs_slice = robs[j:j+c, i].flatten()
# #     plt.hist(robs_slice.tolist())
# #     sns.histplot(robs_slice, kde=True, stat="density")
# # plt.tight_layout()