import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def mse_f(pred, target, *args, **kwargs):
    return (pred - target)**2

def cos_sim_f(pred, target, *args, **kwargs):
    return F.cosine_similarity(pred, target, dim=0)

def corrcoef_f(pred, target, *args, **kwargs):
    return (torch.corrcoef(pred.T) - torch.corrcoef(target.T)) ** 2

def biased_mse_f(pred, target, *args, **kwargs):
    # previously known as SMSE
    return ((pred - target)**2 + target*(1-pred)) * 2

def uncertainty_regularizer(pred):
    #function for reducing uncertainty in predictions
    return 1 - 4*((pred - 0.5) ** 2)

def poisson_f(pred, target, reduce=False, *args, **kwargs):
    return F.poisson_nll_loss(pred, target, log_input=False, full=False, reduction="mean" if reduce else "none")

def binary_cross_entropy_f(pred, target, reduce=False, *args, **kwargs):
    return F.binary_cross_entropy(pred, target, reduction="mean" if reduce else "none")

def f1_f(pred, target, reduce=False, *args, **kwargs):
    eps=1e-8
    tp = pred*target
    fp = (1-target)*pred
    fn = target*(1-pred)
    if reduce:
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2*p*r / (p+r+eps)
    return (1 - f1) / 10

def binary_cross_entropy_with_logits_f(pred, target, reduce=False, *args, **kwargs):
    return F.binary_cross_entropy_with_logits(pred, target, reduction="mean" if reduce else "none")

class BitsPerSpikeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "bits_per_spike"
        self.loss = poisson_f
        self.Rsum = None
        self.Tsum = None
    
    def prepare_loss(self, train_data, cids=None):
        cids = np.arange(len(train_data['robs'])) if cids is None else cids
        self.Rsum = (train_data['robs'][:, cids] * train_data['dfs'][:, cids]).mean(0).detach()
        self.Tsum = train_data['dfs'][:, cids].mean(0).detach()
    
    def forward(self, pred, target, dfs):
        self.Rsum, self.Tsum = self.Rsum.to(pred.device), self.Tsum.to(pred.device)
        LLsum = (self.loss(pred, target, reduce=False) * dfs).mean(0)
        LLneuron = LLsum/self.Rsum
        rbar = self.Rsum/self.Tsum
        LLnulls = torch.log(rbar)-1
        LLneuron = -LLneuron - LLnulls
        return -LLneuron.mean() / np.log(2)

class NDNTLossWrapper(nn.Module):
    '''
        Wrap a loss module to allow NDNT functionality
    '''
    def __init__(self,loss_no_reduction, name, scalable=False):
        """
        Args:
            loss_no_reduction (func): loss function with no reduction, takes arguments
            pred and target both shape (batch, units). Returns loss shaped (batch, units).
            
            name (str): loss name.
            
            scalable (bool, optional): Whether loss gradients remain the same when number 
            of units is changed. Should be turned on whenever neurons dont have a shared core.
        """
        super().__init__()
        self.name = name
        self.loss = loss_no_reduction
        self.scalable = scalable
        
    def forward(self, pred, target, data_filters=1):        
        if self.scalable: # loss gradient magnitude is independent of number of units
            loss_per_neuron = (self.loss(pred, target) * data_filters).sum(0)
            scale_per_neuron = data_filters.sum(0)
            return (loss_per_neuron / scale_per_neuron.clip(1)).sum()
        else: # loss gradient magnitude depends on number of units
            return (self.loss(pred, target) * data_filters).sum()/data_filters.sum().clip(1)
        
class NDNTLossWrapperRobust(nn.Module):
    '''
        Wrap a loss module to allow NDNT functionality
    '''
    def __init__(self,loss_no_reduction, name, scalable=False):
        super().__init__()
        self.name = name
        self.loss = loss_no_reduction
        self.unit_weighting = False
        self.batch_weighting = 0
        self.register_buffer('unit_weights', None)  
        self.register_buffer('av_batch_size', None) 
        self.scalable = scalable

    def set_loss_weighting( self, batch_weighting=None, unit_weighting=None, unit_weights=None, av_batch_size=None ):
        if batch_weighting is not None:
            self.batch_weighting = batch_weighting 
        if unit_weighting is not None:
            self.unit_weighting = unit_weighting 

        if unit_weights is not None:
            self.unit_weights = torch.tensor(unit_weights, dtype=torch.float32)

        assert self.batch_weighting in [-1, 0, 1, 2], "LOSS: Invalid batch_weighting"
        
        if av_batch_size is not None:
            self.av_batch_size = torch.tensor(av_batch_size, dtype=torch.float32)

    def forward(self, pred, target, data_filters=None ):   
        if self.training:     
            alternative_target = (1 - target).clamp(0, 1)
            alternative_loss = pred - alternative_target * torch.log(pred).clamp(min=-6)
            loss = pred - target * torch.log(pred).clamp(min=-6)
            per_neuron_loss = (loss*data_filters).sum(0) / data_filters.sum(0).clamp(min=1)
            per_neuron_alt_loss = (alternative_loss*data_filters).sum(0) / data_filters.sum(0).clamp(min=1)
            # loss = self.loss(pred, target)
            # poisson_loss = pred - target * torch.log(pred).clamp(min=-6)
            # reversed_loss = (target - pred * torch.log(target).clamp(min=-6))
            # per_neuron_loss = (poisson_loss * data_filters).sum(0) / data_filters.sum(0).clamp(min=1)
            # per_neuron_rev_loss = (reversed_loss * data_filters).sum(0) / data_filters.sum(0).clamp(min=1)
            a = 0.99
            per_neuron_loss = a*per_neuron_loss + (1-a)*per_neuron_alt_loss
            # with torch.no_grad():
            #     mn, std = per_neuron_loss.mean(), per_neuron_loss.std()
            #     thresh = mn + 3*std
            #     good_neurons = per_neuron_loss < thresh   
            # loss = (per_neuron_loss*good_neurons).sum()        
            return per_neuron_loss.sum() if self.scalable else per_neuron_loss.mean()
        else:
            if self.scalable:
                return (self.loss(pred, target) * data_filters).sum() / pred.shape[0]
            
            unit_weights = torch.ones( pred.shape[1], device=pred.device)
            if self.batch_weighting == 0:  # batch_size
                unit_weights /= pred.shape[0]
            elif self.batch_weighting == 1: # data_filters
                assert data_filters is not None, "LOSS: batch_weighting requires data filters"
                unit_weights = torch.reciprocal( torch.sum(data_filters, axis=0).clamp(min=1) )
            elif self.batch_weighting == 2: # average_batch_size
                unit_weights /= self.av_batch_size
            # Note can leave as 1s if unnormalized

            if self.unit_weighting:
                unit_weights *= self.unit_weights

            if data_filters is None:
                # Currently this does not apply unit_norms
                loss = self.loss(pred, target).mean()
            else:
                loss_full = self.loss(pred, target)
                # divide by number of valid time points
                loss = torch.sum(torch.mul(unit_weights, torch.mul(loss_full, data_filters))) / len(unit_weights)
            return loss

    def unit_loss(self, pred, target, data_filters=None, temporal_normalize=True ):        
        """This should be equivalent of forward, without sum over units
        Currently only true if batch_weighting = 'data_filter'"""

        if data_filters is None:
            unitloss = torch.sum(
                self.loss(pred, target),
                axis=0)
        else:
            loss_full = self.loss(pred, target)

            unit_weighting = 1.0/torch.maximum(
                torch.sum(data_filters, axis=0),
                torch.tensor(1.0, device=data_filters.device) )

            if temporal_normalize:
                unitloss = torch.mul(unit_weighting, torch.sum( torch.mul(loss_full, data_filters), axis=0) )
            else:
                unitloss = torch.sum( torch.mul(loss_full, data_filters), axis=0 )
        return unitloss
        # END PoissonLoss_datafilter.unit_loss
        
class Poisson(NDNTLossWrapper):
    def __init__(self, scalable=False):
        super().__init__(poisson_f, "poisson", scalable=scalable)
class PoissonRobust(NDNTLossWrapperRobust):
    def __init__(self, scalable=False):
        super().__init__(poisson_f, "poisson", scalable=scalable)