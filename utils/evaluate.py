import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class EvalModule(nn.Module):
    """Module that evaluates the log-likelihood of a model on a given dataset.
    This is used for training.
    """
    def __init__(self, cids, meta_cids=None):
        super().__init__()
        self.cids = cids
        self.meta_cids = meta_cids
        self.LLnull, self.LLsum, self.nspikes = 0, 0, 0
    def reset(self):
        self.LLnull, self.LLsum, self.nspikes = 0, 0, 0
    def startDS(self, train_ds=None, means=None):
        if means is not None:
            self.register_buffer("mean_spikes", means[self.meta_cids])
        elif train_ds is not None:
            if hasattr(train_ds, 'covariates'):
                print("Using train_ds to calculate mean spikes")
                sum_spikes = (train_ds.covariates["robs"][:, self.cids] * train_ds.covariates["dfs"][:, self.cids]).sum(dim=0)
                total_valid = train_ds.covariates["dfs"][:, self.cids].sum(dim=0)
            else:
                device = next(iter(train_ds))["robs"].device
                sum_spikes = torch.zeros(len(self.cids), device=device)
                total_valid = torch.zeros(len(self.cids), device=device)
                for b in tqdm(train_ds):
                    sum_spikes = sum_spikes + (b["dfs"][:, self.cids] * b["robs"][:, self.cids]).sum(dim=0)
                    total_valid = total_valid + b["dfs"][:, self.cids].sum(dim=0)
            self.register_buffer("mean_spikes", (sum_spikes / total_valid)[self.meta_cids])
        else:
            raise ValueError("Either train_ds or means must be provided.")
    def getLL(self, pred, batch):
        poisson_ll = batch["robs"][:, self.cids][:, self.meta_cids] * torch.log(pred[:, self.meta_cids] + 1e-8) - pred[:, self.meta_cids]
        return (poisson_ll * batch["dfs"][:, self.cids][:, self.meta_cids]).sum(dim=0)
    def __call__(self, rpred, batch):
        if not hasattr(self, 'mean_spikes'):
            print("(ignore if just sanity checking) mean_spikes not initialized. Call startDS first.")
            return
        llsum = self.getLL(rpred, batch).cpu()
        llnull = self.getLL(self.mean_spikes.to(rpred.device).expand(*rpred[:, self.meta_cids].shape), batch).cpu()
        self.LLnull = self.LLnull + llnull
        self.LLsum = self.LLsum + llsum
        self.nspikes = self.nspikes + (batch["dfs"][:, self.cids][:, self.meta_cids] * batch["robs"][:, self.cids][:, self.meta_cids]).sum(dim=0).cpu()
        del llsum, llnull, rpred, batch
    def closure(self):
        if type(self.LLnull) is int:
            print("(ignore if just sanity checking) EvalModule has not been called yet")
            return torch.tensor(0.0)
        zeros = torch.where((self.LLsum == 0))[0].tolist() #check for zeros
        if zeros:
            print(f"(ignore if just sanity checking) no spikes detected for neurons {zeros}. Check your cids, data and datafilters.")
            self.reset()
            return torch.tensor(0.0)
        bps = (self.LLsum - self.LLnull)/self.nspikes.clamp(1)/np.log(2)
        self.reset()
        return bps
    
class Evaluator(EvalModule):
    """(synonym for EvalModule) Module that evaluates the log-likelihood of a model on a given dataset.
    """
    pass
    