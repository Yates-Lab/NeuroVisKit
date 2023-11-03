
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
        elif type(index) is torch.Tensor or type(index) is np.ndarray:
            return self.__getitem__(index.tolist())
        else:
            # takes care of slices
            relevant_blocks = self.block[index]

        # unravels starts and stops for each block
        inds = [i for block in relevant_blocks for i in range(*block)]

        # calling the super class returns a dictionary of tensors
        return super().__getitem__(inds)
    
    def dsFromInds(self, inds, in_place=False, safe=True):
        if in_place:
            return ContiguousDataset(self.covariates, [self.block[i] for i in inds])
        if safe:
            def concat_dicts(*ds):
                return {k: torch.cat([d[k] for d in ds], dim=0) for k in ds[0].keys()}
            data, blocks = [], []
            block = 0
            for i in inds:
                data.append(self[i])
                block_len = self.block[i][1] - self.block[i][0]
                blocks.append((block, block + block_len))
                block += block_len
            return ContiguousDataset(concat_dicts(*data), blocks)
        raise NotImplementedError("Not implemented yet. please set safe=true")
        new_covariates = self[inds]
        blocks = [self.block[i] for i in range(len(self.block)) if i in inds]
        new_blocks = []
        bstart = 0
        for b in blocks:
            bstop = bstart + (b[1] - b[0])
            new_blocks.append((bstart, bstop))
            bstart = bstop
        return ContiguousDataset(new_covariates, new_blocks)
    
    @staticmethod
    def combine_contiguous_ds(datasets):
        """Combine multiple datasets into one"""
        ds = datasets[0]
        blocks = list(ds.block)
        data = {
            k: [v] for k, v in ds.covariates.items()
        }
        for d in datasets[1:]:
            block_shift = blocks[-1][-1]
            for b in d.block:
                blocks.append((b[0] + block_shift, b[1] + block_shift))
            for k, v in d.covariates.items():
                data[k].append(v)
        for k, v in data.items():
            data[k] = torch.cat(v, dim=0)
        out = ContiguousDataset(data, blocks)
        if hasattr(datasets[0], 'stim_index'):
            setattr(out, 'stim_index', np.concatenate([d.stim_index for d in datasets]))
        if hasattr(datasets[0], 'requested_stims'):
            setattr(out, 'requested_stims', datasets[0].requested_stims)
        return out
    
    @staticmethod
    def get_stim_indices(self, stim_name='Gabor'):
        if isinstance(stim_name, str):
            stim_name = [stim_name]
        stim_id = [i for i,s in enumerate(self.requested_stims) if s in stim_name]
        return np.where(np.isin(self.stim_index, stim_id))[0]
    @staticmethod
    def get_stim_counts(self):
        counts = np.unique(self.stim_index, return_counts=True)[1]
        stims = self.requested_stims
        return {s: c for s, c in zip(stims, counts)}

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
        
def BlockedDataLoader(dataset, inds=None, batch_size=1):
    '''
    Creates a dataloader that returns contiguous blocks of data from a dataset.
    Each block includes multiple samples, here "batch_size" operates on the block level and
    the returned batches will NOT necessarily have the same size
    '''
    from torch.utils.data import DataLoader
    if inds is None:
        inds = list(range(len(dataset)))

    sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SubsetRandomSampler(inds),
                batch_size=batch_size,
                drop_last=False)

    if dataset[0]['stim'].device.type == 'cuda':
        num_workers = 0
    else:
        import os
        num_workers = os.cpu_count()//2

    dl = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=num_workers)
    return dl