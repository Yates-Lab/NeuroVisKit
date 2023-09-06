
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
import NeuroVisKit.utils.utils as utils

class GenericDataset(Dataset):
    '''
    Generic Dataset can be used to create a quick pytorch dataset from a dictionary of tensors
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
    '''
    def __init__(self,
        data):

        self.covariates = data
        self.requested_covariates = list(self.covariates.keys())

    def to(self, device):
        self.covariates = utils.to_device(self.covariates, device)
        return self
        
    def __len__(self):

        return self.covariates['stim'].shape[0]

    def __getitem__(self, index):
        return {cov: self.covariates[cov][index,...] for cov in self.requested_covariates}

# TODO: Blocked dataloader

class ContiguousDataset(GenericDataset):
    '''
    Contiguous Dataset creates a pytorch dataset from a dictionary of tensors that serves contiguous blocks
    Called the same way as GenericDataset, but with an additional "blocks" argument
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
        Blocks: List of tuples. Each tuple is a start and stop index for a block of contiguous data.
    '''

    def __init__(self, data, blocks):
        
        super().__init__(data)

        self.block = blocks
    
    def __len__(self):

        return len(self.block)

    def __getitem__(self, index):

        if type(index) is int:
            relevant_blocks = [self.block[index]]
        elif type(index) is list:
            relevant_blocks = [self.block[i] for i in index]
        else:
            # takes care of slices
            relevant_blocks = self.block[index]

        # unravels starts and stops for each block
        inds = [i for block in relevant_blocks for i in range(*block)]

        # calling the super class returns a dictionary of tensors
        return super().__getitem__(inds)
    
    @staticmethod
    def fromDataset(ds, inds=None):

        if inds is None:
            inds = list(range(len(ds)))

        blocks = []
        stim = []
        robs = []
        eyepos = []
        dfs = []
        bstart = 0
        print("building dataset")
        for ii in tqdm(inds):
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
        return ContiguousDataset(d, blocks)
        