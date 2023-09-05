
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm

class GenericDataset(Dataset):
    '''
    Generic Dataset can be used to create a quick pytorch dataset from a dictionary of tensors
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
        device: Device to put each tensor on. Default is cpu.
    '''
    def __init__(self,
        data,
        dtype=torch.float32,
        device=None):

        self.covariates = {}
        for cov in list(data.keys()):
            self.covariates[cov] = data[cov].to(dtype)

        if device is None:
            device = torch.device('cpu')
        
        self.device = device
        
        try:
            if 'stim' in self.covariates.keys() and len(self.covariates['stim'].shape) > 3:
                self.covariates['stim'] = self.covariates['stim'].contiguous(memory_format=torch.channels_last)
        except:
            pass

        self.cov_list = list(self.covariates.keys())
        for cov in self.cov_list:
            self.covariates[cov] = self.covariates[cov].to(self.device)
        
    def __len__(self):

        return self.covariates['stim'].shape[0]

    def __getitem__(self, index):
        return {cov: self.covariates[cov][index,...] for cov in self.cov_list}


class ContiguousDataset(GenericDataset):
    '''
    Contiguous Dataset creates a pytorch dataset from a dictionary of tensors that serves contiguous blocks
    Called the same way as GenericDataset, but with an additional "blocks" argument
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
        device: Device to put each tensor on. Default is cpu.
    '''

    def __init__(self, data, blocks, dtype=torch.float32, device=None):
        
        super().__init__(data, dtype, device)

        self.blocks = blocks
    
    def __len__(self):

        return len(self.blocks)

    def __getitem__(self, index):

        if isinstance(index, int) or isinstance(index, np.int64):
            index = [index]
        elif isinstance(index, slice):
            index = np.arange(index.start or 0, index.stop or len(self.blocks), index.step or 1)

        inds = []
        for i in index:
            inds.append(torch.arange(self.blocks[i][0], self.blocks[i][1])) #, device=self.device
        
        inds = torch.cat(inds)

        return {cov: self.covariates[cov][inds,...] for cov in self.cov_list}
    
    def fromDataset(ds, dtype=torch.float32, device=None):
        blocks = []
        stim = []
        robs = []
        eyepos = []
        dfs = []
        bstart = 0
        print("building dataset")
        for ii in tqdm(range(len(ds))):
            batch = ds[ii]
            stim.append(batch['stim'])
            robs.append(batch['robs'])
            eyepos.append(batch['eyepos'])
            dfs.append(batch['dfs'])
            bstop = bstart + batch['stim'].shape[0]
            blocks.append((bstart, bstop))
            bstart = bstop
        stim = torch.cat(stim, dim=0)
        robs = torch.cat(robs, dim=0)
        eyepos = torch.cat(eyepos, dim=0)
        dfs = torch.cat(dfs, dim=0)
        d = {
            "stim": stim,
            "robs": robs,
            "eyepos": eyepos,
            "dfs": dfs,
        }
        return ContiguousDataset(d, blocks, dtype=dtype, device=device)
        