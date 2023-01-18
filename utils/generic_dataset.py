
import torch
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    '''
    Generic Dataset can be used to create a quick pytorch dataset from a dictionary of tensors
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
        device: Device to put each tensor on. Default is cpu.
    '''
    def __init__(self,
        data,
        device=None):

        self.covariates = {}
        for cov in list(data.keys()):
            self.covariates[cov] = data[cov]

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