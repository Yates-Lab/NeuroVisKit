import torch
import NDNT
import torch.nn as nn
import torch.nn.functional as F

def get_prior(device='cpu'):
    prior = [0.0205, 0.0094, 0.0211, 0.0256, 0.044, 0.0276, 0.0335, 0.0319, 0.0312, 0.0412, 0.022, 0.0397, 0.0307, 0.0306, 0.0297, 0.0143, 0.0259, 0.0374, 0.0386, 0.0146, 0.0226, 0.0266, 0.0089, 0.0179, 0.0354, 0.0374, 0.0295, 0.011, 0.0227, 0.0195, 0.015, 0.0458, 0.0218, 0.0342, 0.037, 0.0833, 0.0573, 0.0263, 0.0173, 0.0365, 0.0405, 0.0143, 0.0192, 0.0396, 0.0169, 0.02, 0.027, 0.0304, 0.0205, 0.0426, 0.0232, 0.033, 0.0181, 0.0218, 0.02, 0.0202, 0.033, 0.022, 0.0267, 0.013, 0.0145, 0.1243]
    return torch.tensor(prior, device=device).unsqueeze(0)

def mse_f(pred, target, *args, **kwargs):
    return (pred - target)**2

def cos_sim_f(pred, target, *args, **kwargs):
    return F.cosine_similarity(pred, target, dim=0)

def corrcoef_f(pred, target, *args, **kwargs):
    return (torch.corrcoef(pred.T) - torch.corrcoef(target.T)) ** 2

def biased_mse_f(pred, target, *args, **kwargs):
    # previously known as SMSE
    return ((pred - target)**2 + target*(1-pred)) * 2

def balanced_biased_mse_f(pred, target, *args, **kwargs):
    # previously known as SMSE2 (except now with prior)
    inds0 = target==0
    w = get_prior(device=target.device).broadcast_to(target.shape).clone()
    w[inds0] = (pred[inds0] - target[inds0])**2 / (1-w[inds0])
    w[~inds0] = ((pred[~inds0] - target[~inds0])**2 + (1 - pred[~inds0])**2) / w[~inds0]
    return w * 3

def uncertainty_regularizer(pred):
    #function for reducing uncertainty in predictions
    return 1 - 4*((pred - 0.5) ** 2)

def poisson_f(pred, target, reduce=False, *args, **kwargs):
    return F.poisson_nll_loss(pred, target, log_input=False, full=False, reduction="mean" if reduce else "none")

def binary_cross_entropy_f(pred, target, reduce=False, *args, **kwargs):
    return F.binary_cross_entropy(pred, target, reduction="mean" if reduce else "none")

def balanced_binary_cross_entropy_f(pred, target, reduce=False, *args, **kwargs):
    w = get_prior(device=target.device).broadcast_to(target.shape).clone()
    return w * F.binary_cross_entropy(pred, target, reduction="mean" if reduce else "none") / 10
    
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

def balanced_binary_cross_entropy_with_logits_f(pred, target, reduce=False, *args, **kwargs):
    w = get_prior(device=target.device).broadcast_to(target.shape).clone()
    inds0 = target == 0
    w[inds0] = 1 / (1-w[inds0])
    w[~inds0] = 1 / w[~inds0]
    return w * F.binary_cross_entropy_with_logits(
        pred, target, weight=w, reduction="mean" if reduce else "none"
    ) / 10

class DatafilterLossWrapper(nn.Module):
    '''
        Wrap a loss function to be used as a module, with data filters.
    '''
    def __init__(self, loss_no_reduction, name):
        super().__init__()
        self.name = name
        self.loss = loss_no_reduction
    def forward(self, pred, target, data_filters=None):    
        eps = 1e-8
        loss = self.loss(pred, target)    
        if data_filters is None:
            return loss.mean()
        else:
            return ((loss * data_filters).sum(0) / data_filters.sum(0).clamp(min=1)).mean()
        
class LossFuncWrapper(nn.Module):
    '''
        Wrap a loss function to be used as a module.
    '''
    def __init__(self, lossF_no_reduction, name):
        super().__init__()
        self.loss = lossF_no_reduction
        self.name = name
    def forward(self, pred, target, *args, **kwargs):
        return self.loss(pred, target, reduce=True).mean()
    
class NDNTLossWrapper(nn.Module):
    '''
        Wrap a loss module to allow NDNT functionality
    '''
    def __init__(self,loss_no_reduction, name):
        super().__init__()
        self.name = name
        self.loss = loss_no_reduction
        self.unit_weighting = False
        self.batch_weighting = 0
        self.register_buffer('unit_weights', None)  
        self.register_buffer('av_batch_size', None) 

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
    # END PoissonLoss_datafilter.forward

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
        
class ExperimentalNDNTLossWrapper(nn.Module):
    '''
        Wrap a loss module to allow NDNT functionality
    '''
    def __init__(self,loss_no_reduction, name):
        super().__init__()
        self.name = name
        self.loss = loss_no_reduction
        self.unit_weighting = True
        self.batch_weighting = 1
        prior = get_prior().squeeze()
        self.register_buffer('unit_weights', 1/prior/len(prior))  
        self.register_buffer('av_batch_size', None) 

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
            unit_weights *= self.unit_weights.to(pred.device)

        if data_filters is None:
            # Currently this does not apply unit_norms
            loss = self.loss(pred, target).mean()
        else:
            loss_full = self.loss(pred, target)
            # divide by number of valid time points
            
            loss = torch.sum(torch.mul(unit_weights, torch.mul(loss_full, data_filters))) / len(unit_weights)
        return loss
    # END PoissonLoss_datafilter.forward

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
    
LOSS_DICT = {
    'mse': NDNTLossWrapper(mse_f, "mse"),
    'poisson': NDNTLossWrapper(poisson_f, "poisson"),
    'pearson': LossFuncWrapper(corrcoef_f, "pearson"),
    'smse': NDNTLossWrapper(biased_mse_f, "smse"),
    'bsmse': NDNTLossWrapper(balanced_biased_mse_f, "bsmse"),
    'f1': NDNTLossWrapper(f1_f, "f1"),
    'ce': NDNTLossWrapper(binary_cross_entropy_f, "ce"),
    'bce': NDNTLossWrapper(balanced_binary_cross_entropy_f, "bce"),
    'ce_logits': NDNTLossWrapper(binary_cross_entropy_with_logits_f, "ce_logits"),
    'bce_logits': NDNTLossWrapper(balanced_binary_cross_entropy_with_logits_f, "bce_logits"),
    'torch_poisson': LossFuncWrapper(poisson_f, "torch_poisson"),
    'experimental': ExperimentalNDNTLossWrapper(poisson_f, "experimental"),
}

NONLINEARITY_DICT = {
    **{k: nn.Softplus() for k in ['poisson', 'pearson', 'torch_poisson', 'experimental']},
    **{k: nn.Sigmoid() for k in ['ce', 'bce', 'f1']},
    **{k: nn.Identity() for k in ['mse', 'smse', 'bsmse', 'ce_logits', 'bce_logits']},
}

def get_loss(config):
    return LOSS_DICT[config['loss']], NONLINEARITY_DICT[config['loss']]


# class robust_loss():
#     def __init__(self):
#         self.ctx
#     def forward(self, t, c):
#         self.ctx = (t, torch.tensor([c]))
#         inds = torch.abs(t)<c
#         t[inds] = torch.abs(t[inds])
#         t[~inds] = (t[~inds] ** 2)/c
#         return t.mean()
#     def backward(self, grad_output):
#         t, c = ctx.saved_tensors
#         inds = torch.abs(t)<c
#         grad_input = grad_output.clone()
#         grad_input[inds] = grad_input[inds] * torch.sign(t[inds])
#         grad_input[~inds] = grad_input[~inds] * 2 * t[~inds] / c
#         return grad_input, None

# def robust_loss_grad(t, c):
#     t = t.clone()
#     inds = torch.abs(t)<c
#     t[inds] = torch.sign(t[inds])
#     t[~inds] = 2 * t[~inds] / c
#     return t

# class znorm(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, batch_n, mean, std):
#         ctx.save_for_backward(x, torch.tensor([batch_n]), mean, std)
#         return (x - mean) / std

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, batch_n, mean, std = ctx.saved_tensors
#         stdx, meanx = x.std(0), x.mean(0)        
#         dx = grad_output / std
#         dm = -grad_output / std
#         ds = -grad_output * (x - mean) / std**2
        
#         dm1 = -robust_loss_grad(meanx.unsqueeze(0) - mean, 0.01)
#         ds1 = -robust_loss_grad(stdx.unsqueeze(0) - std, 0.01)
#         dlddz = robust_loss_grad((x - mean) / std - (x - meanx) / stdx, 0.01) 
#         dldx = dlddz / std - dlddz / stdx
        
#         # w = torch.sigmoid(batch_n/1000 - 10).item()
#         # wx = (dldx**2).mean(0) /  (dx**2).mean(0) * 1
#         c = 0.1
#         wm = (dm**2).mean(0) /  (dm1**2).mean(0) * c
#         ws = (ds**2).mean(0) /  (ds1**2).mean(0) * c
#         return dx, None, dm*(1-c) + dm1*wm , ds*(1-c) + ds1*ws

# class ZNorm(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.mean = nn.Parameter(torch.zeros(1, dims))
#         self.std = nn.Parameter(torch.ones(1, dims))
#         self.batch = -1
#     def forward(self, x):
#         self.batch += 1
#         return znorm.apply(x, self.batch, self.mean, self.std)
    # def backward(self, grad_output):
        #     x = self.cache
        #     stdx, meanx = x.std(0), x.mean(0)
        #     tzx = (x - meanx) / stdx
        #     zx = (x - self.mean) / torch.exp(self.std)
        #     dnorm = torch.abs(torch.stack([tzx, zx])).max(0)
        #     dx = (tzx - zx) / dnorm
            
        #     stdx = torch.log(stdx)
        #     stdx[stdx.isnan()] = -10
        #     mean_norm = torch.abs(torch.stack([self.mean.squeeze(0), meanx])).max(0)
        #     std_norm = torch.abs(torch.stack([self.std.squeeze(0), stdx])).max(0)
        #     grad_mean = (meanx - self.mean.squeeze(0)) / mean_norm
        #     std_grad = (stdx - self.std.squeeze(0)) / std_norm
            
        #     dx2 = grad_output * torch.exp(self.std) + self.mean
        #     w = torch.sigmoid(self.batch/1000 - 10)
        #     # wx = torch.exp(-((self.batch-5000)/2000)**2)
        #     wx = (dx2**2).mean(0) /  (dx**2).mean(0) * 0.1
        #     return dx*wx + dx2, grad_mean*w, std_grad*w