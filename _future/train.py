import torch
import dill
from NeuroVisKit.utils.trainer import train
# from models import ModelWrapper, CNNdense
from NeuroVisKit._utils.utils import seed_everything
from dadaptation import DAdaptAdam, DAdaptSGD, DAdaptAdaGrad
from dog import DoG, LDoG
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

class InMemoryContiguousDataset(torch.utils.data.Dataset):
    def __init__(self, ds, inds=None):
        super().__init__()
        self.ds = ds
        self.inds = inds or list(range(len(ds)))
        self.block = []
        self.dictionary = self.build_dataset()
    def preload(self, device):
        for k, v in self.dictionary.items():
            self.dictionary[k] = v.to(device)
    def __getitem__(self, index):
        if type(index) is int:
            relevant_blocks = [self.block[index]]
        elif type(index) is list:
            relevant_blocks = [self.block[i] for i in index]
        else:
            relevant_blocks = self.block[index]
        inds = [i for block in relevant_blocks for i in range(*block)]
        return {k: v[inds] for k, v in self.dictionary.items()}
    def __len__(self):
        return len(self.block)
    def build_dataset(self):
        stim = []
        robs = []
        eyepos = []
        dfs = []
        bstart = 0
        print("building dataset")
        for ii in tqdm(self.inds):
            batch = self.ds[ii]
            stim.append(batch['stim'])
            robs.append(batch['robs'])
            eyepos.append(batch['eyepos'])
            dfs.append(batch['dfs'])
            bstop = bstart + batch['stim'].shape[0]
            self.block.append((bstart, bstop))
            bstart = bstop
        stim = torch.cat(stim, dim=0)
        robs = torch.cat(robs, dim=0)
        eyepos = torch.cat(eyepos, dim=0)
        dfs = torch.cat(dfs, dim=0)
        return {
            "stim": stim,
            "robs": robs,
            "eyepos": eyepos,
            "dfs": dfs,
        }
        
class InMemoryContiguousDataset2(torch.utils.data.Dataset):
    def __init__(self, ds, inds=None):
        super().__init__()
        self.ds = ds
        self.inds = inds or list(range(len(ds)))
        self.list = []
        self.build_dataset()
    def preload(self, device):
        for batch in self.list:
            for k, v in batch.items():
                batch[k] = v.to(device)
    def __getitem__(self, index):
        if type(index) is int:
            return self.list[index]
        elif type(index) is slice:
            items = self.list[index]
        else:
            items = [self.list[i] for i in index]
        return {k: torch.cat([d[k] for d in items], dim=0) for k in items[0].keys()}
    def __len__(self):
        return len(self.list)
    def build_dataset(self):
        print("building dataset")
        for ii in tqdm(self.inds):
            batch = self.ds[ii]
            self.list.append(batch)
            
class InMemoryContiguousDataset3(TensorDataset):
    def __init__(self, ds, inds=None):
        self.inds = inds if inds is not None else list(range(len(ds)))
        self.block = []
        dictionary = self.build_dataset(ds)
        self.keys = list(dictionary.keys())
        super().__init__(*dictionary.values())
        print(len(self.block), len(self.inds))
    def preload(self, device):
        self.tensors = [t.to(device) for t in self.tensors]
    def __getitem__(self, index):
        if type(index) is int:
            relevant_blocks = [self.block[index]]
        elif type(index) is list:
            relevant_blocks = [self.block[i] for i in index]
        else:
            relevant_blocks = self.block[index]
        inds = [i for block in relevant_blocks for i in range(*block)]
        values = super().__getitem__(inds)
        return {k: v for k, v in zip(self.keys, values)}
    def __len__(self):
        return len(self.block)
    def build_dataset(self, ds):
        stim = []
        robs = []
        eyepos = []
        dfs = []
        bstart = 0
        print("building dataset")
        for ii in tqdm(self.inds):
            batch = ds[ii]
            stim.append(batch['stim'])
            robs.append(batch['robs'])
            eyepos.append(batch['eyepos'])
            dfs.append(batch['dfs'])
            bstop = bstart + batch['stim'].shape[0]
            self.block.append((bstart, bstop))
            bstart = bstop
        stim = torch.cat(stim, dim=0)
        robs = torch.cat(robs, dim=0)
        eyepos = torch.cat(eyepos, dim=0)
        dfs = torch.cat(dfs, dim=0)
        return {
            "stim": stim,
            "robs": robs,
            "eyepos": eyepos,
            "dfs": dfs,
        }

    
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
    def train_f(model, train_loader, val_loader, checkpoint_dir, device, patience=40, memory_saver=False):
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
            patience=patience,
            memory_saver=memory_saver)
        return val_loss_min
    return train_f

TRAINER_DICT = {
    'adam': train_f(torch.optim.Adam, lr=0.001),
    'adam1e-4': train_f(torch.optim.Adam, lr=0.0001),
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