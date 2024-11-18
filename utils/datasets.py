
from random import shuffle
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
import NeuroVisKit.utils.utils as utils
import torch.nn.functional as F

class GenericDataset(Dataset):
    '''
    Generic Dataset can be used to create a quick pytorch dataset from a dictionary of tensors
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
    '''
    def __init__(self,
        data):
        self.covariates = data
    
    @property
    def requested_covariates(self):
        return list(self.covariates.keys())
    
    def to(self, device):
        self.covariates = utils.to_device(self.covariates, device)
        return self
        
    def __len__(self):
        return next(iter(self.covariates.values())).shape[0]

    def __getitem__(self, index):
        return {cov: self.covariates[cov][index,...] for cov in self.requested_covariates}

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
        # calling the super class returns a dictionary of tensors
        return super().__getitem__(self.blockIndsToInds(index))
    
    def blockIndsToInds(self, index):
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
        return inds
    
    def dsFromInds(self, inds, in_place=False, safe=True):
        if in_place:
            return ContiguousDataset(self.covariates, [self.block[i] for i in inds])
        if safe:
            def concat_dicts(*ds):
                return {k: torch.cat([d[k] for d in ds], dim=0) for k in ds[0].keys()}
            data, blocks = [], []
            block = 0
            for i in inds:
                data.append(self[int(i)])
                block_len = self.block[i][1] - self.block[i][0]
                blocks.append((block, block + block_len))
                block += block_len
            out = ContiguousDataset(concat_dicts(*data), blocks)
            if hasattr(self, 'stim_index'):
                setattr(out, 'stim_index', self.stim_index[inds])
            if hasattr(self, 'requested_stims'):
                setattr(out, 'requested_stims', self.requested_stims)
            return out
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
    
class BlockAdaptiveSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, block_sizes, min_block_size=500, inds=None):
        if inds is not None:
            self.block_inds = inds
            assert len(self.block_inds) == len(block_sizes), "block_sizes and inds must have the same length"
        else:
            self.block_inds = np.arange(len(block_sizes)) 
        self.block_sizes = block_sizes
        self.ind_to_size = {i: s for i, s in zip(self.block_inds, self.block_sizes)}
        self.min_block_size = min_block_size
        self.is_reset = False
        self.reset()
    def test(self):
        import matplotlib.pyplot as plt
        blocks = []
        block_sizes = []
        for b in self.blocks:
            blocks.extend(b)
            block_sizes.append(sum([self.ind_to_size[i] for i in b]))
        for i in self.block_inds:
            assert i in blocks, f"{i} not in blocks"
        plt.hist(block_sizes)
        return block_sizes
    def reset(self):
        inds = np.random.permutation(len(self.block_sizes))
        shuffled_block_sizes = self.block_sizes[inds]
        shuffled_block_inds = self.block_inds[inds]
        #go from block_ind_to_counter
        blocks = []
        ctr = 0
        while ctr < len(shuffled_block_sizes):
            block_size = 0
            block = []
            while block_size < self.min_block_size and ctr < len(shuffled_block_sizes):
                block_size += shuffled_block_sizes[ctr]
                block.append(int(shuffled_block_inds[ctr]))
                ctr += 1
            blocks.append(block)
        if block_size < self.min_block_size:
            last_block = blocks.pop()
            smallest_block_inds = np.argsort([sum([self.ind_to_size[i] for i in b]) for b in blocks])
            for i in range(len(last_block)):
                blocks[smallest_block_inds[i]].append(last_block[i])
        self.blocks = blocks
        self.is_reset = True
    def __iter__(self):
        if not self.is_reset:
            self.reset()
        self.is_reset = False
        for b in self.blocks:
            yield b
    def __len__(self):
        return len(self.blocks)

def BlockedAdaptiveDataLoader(dataset, inds=None, min_block_size=500, cpu_num_workers=0.5, num_lags=None):
    from torch.utils.data import DataLoader
    assert hasattr(dataset, 'block'), "Dataset must have block attribute"
    block_sizes = np.diff(np.array(dataset.block),1)[:,0] - (num_lags-1 or 0)
    if inds is None:
        inds = np.array(list(range(len(dataset))))
    sampler = BlockAdaptiveSampler(block_sizes[inds], min_block_size, inds)
    if dataset.covariates['stim'].device.type == 'cuda':
        num_workers = 0
    else:
        import os
        num_workers = int(os.cpu_count() * cpu_num_workers)
    dl = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=num_workers)
    return dl
        
def BlockedDataLoader(dataset, inds=None, batch_size=1, cpu_num_workers=0.5):
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
        num_workers = int(os.cpu_count() * cpu_num_workers)

    dl = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=num_workers)
    return dl

def pad_first_dim(t, pad_size):
    # pad_size is a tuple of (left, right), pad the first dimension of tensor t that has shape (N, ...)
    padding_dims = pad_size
    for _ in range(1, t.dim()):
        padding_dims = (0, 0) + padding_dims
    return F.pad(t, padding_dims)
class DSAutoDFS(ContiguousDataset):
    """
        Dataset with automatic (and dynamic) data filter generation.
        
        data is a dictionary of tensors.
        blocks is a list of tuples, each tuple is a start and stop index for a block of contiguous data.
        num_lags is the number of lags to filtered out in the beginning of each block.
        min_blocksize is the minimum size of each block.
        
    """
    def __init__(self, data, blocks, num_lags, min_blocksize=500, block_dfs_start=0):
        super().__init__(data, blocks)
        self.num_lags = num_lags
        self.min_blocksize = min_blocksize
        if not torch.is_tensor(block_dfs_start):
            block_dfs_start = torch.ones(num_lags-1) * block_dfs_start
        self.block_dfs_start = block_dfs_start
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            #convert slice to list
            idx = list(range(*idx.indices(len(self))))
        elif not hasattr(idx, "__iter__"):
            if hasattr(idx, "item"):
                idx = idx.item()
            return self.dfsfy(super().__getitem__(idx))
        return self.collate([self[i] for i in idx])
    def dfsfy(self, batch):
        batch["dfs"] = batch["dfs"] if "dfs" in batch else torch.ones_like(batch["robs"])
        batch["dfs"][:self.num_lags-1] = batch["dfs"][:self.num_lags-1] * self.block_dfs_start.to(batch["dfs"].device)[:, None]
        if len(batch["robs"]) < self.min_blocksize:
            p = self.min_blocksize - len(batch["robs"])
            for k in batch.keys():
                batch[k] = pad_first_dim(batch[k], (0, p))
        return batch
    def collate(self, batches):
        d = {}
        for k in batches[0].keys():
            d[k] = torch.concat([b[k] for b in batches])  
        return d
    
def split_blocks(ds, max_block_size=500, num_lags=60, inds=None):
    blocks_sizes = np.diff(np.array(ds.block),1)[:,0]
    inds = list(range(len(ds.block)))
    for i in inds:
        if blocks_sizes[i] > max_block_size:
            s, e = ds.block[i]
            snew, enew = s, min(s + max_block_size, e)
            assert enew - snew <= max_block_size
            ds.block[i] = [snew, enew]
            # snew = s + max_block_size - num_lags + 1
            while enew < e:
                snew = snew + max_block_size - num_lags + 1
                enew = min(snew + max_block_size, e)
                assert enew - snew <= max_block_size
                ds.block.append([snew, enew])
    return ds

def split_blocks_old(ds, train_inds, val_inds, max_block_size=500, num_lags=60):
    blocks_sizes = np.diff(np.array(ds.block),1)[:,0]
    for inds in [train_inds]:
        for ind in range(len(inds)):
            i = inds[ind]
            if blocks_sizes[i] > max_block_size:
                s, e = ds.block[i]
                snew, enew = s, min(s + max_block_size, e)
                assert enew - snew <= max_block_size
                ds.block[i] = [snew, enew]
                # snew = s + max_block_size - num_lags + 1
                while enew < e:
                    snew = snew + max_block_size - num_lags + 1
                    enew = min(snew + max_block_size, e)
                    assert enew - snew <= max_block_size
                    ds.block.append([snew, enew])
                    inds.append(len(ds.block) - 1)
    return ds, np.array(train_inds, dtype=np.int64), np.array(val_inds, dtype=np.int64)

def remove_small_blocks(ds, min_block_size=20):
    blocks = np.diff(np.array(ds.block),1)[:,0]
    inds = np.where(blocks >= min_block_size)[0]
    return ds.dsFromInds(inds)

def train_val_split(ds, train_size=0.8, seed=0):
    np.random.seed(seed)
    blocks = np.diff(np.array(ds.block),1)[:,0]
    perm = np.random.permutation(len(blocks))
    train_inds = []
    train_len, total_len = 0, blocks.sum()
    i = 0
    while train_len < train_size * total_len and i < len(perm):
        j = perm[i]
        train_inds.append(j)
        train_len += blocks[j]
        i += 1
    print(f"achieved {train_len/sum(blocks):.3f}")
    val_inds = np.setdiff1d(np.arange(len(blocks)), train_inds)
    return ds.dsFromInds(train_inds), ds.dsFromInds(val_inds)