import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper for loss functions.
class LossWrapper(nn.Module):
    '''
        Wrap a loss module to allow added functionality of datafilters and proper scaling.
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

# Poisson negative log likelihood loss.
class Poisson(LossWrapper):
    def __init__(self, scalable=False):
        super().__init__(self.functional, "Poisson", scalable)
    @staticmethod
    def functional(pred, target, *args, **kwargs):
        return pred - target * torch.log(pred + 1e-8)

# Mean squared error loss.
class MSE(LossWrapper):
    def __init__(self, scalable=False):
        super().__init__(self.functional, "MSE", scalable)
    @staticmethod
    def functional(pred, target, *args, **kwargs):
        return (pred - target)**2

# Binary cross entropy loss. Clips values to [0, 1] range.
class BCE(LossWrapper):
    def __init__(self, scalable=False):
        super().__init__(self.functional, "BCE", scalable)
    @staticmethod
    def functional(pred, target, *args, **kwargs):
        return F.binary_cross_entropy(pred.clip(0, 1), target.clip(0, 1), reduction="none")