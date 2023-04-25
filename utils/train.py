import torch
import dill
import NDNT
from utils.trainer import train
from models import ModelWrapper, CNNdense
from utils.get_models import get_cnn
from utils.utils import seed_everything
from dadaptation import DAdaptAdam, DAdaptSGD, DAdaptAdaGrad
from dog import DoG, LDoG
import torch.nn as nn
import torch.nn.functional as F

def train_f(opt, **kwargs):
    def train_f(model, train_loader, val_loader, checkpoint_dir, device, patience=30):
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
            patience=patience)
        return val_loss_min
    return train_f

TRAINER_DICT = {
    'adam': train_f(torch.optim.Adam, lr=0.001),
    # 'lbfgs': train_lbfgs,
    'dadaptadam': train_f(DAdaptAdam, lr=1),
    'dadaptsgd': train_f(DAdaptSGD, lr=1),
    'dadaptadagrad': train_f(DAdaptAdaGrad, lr=1),
    'dog': train_f(DoG, lr=1),
    'ldog': train_f(LDoG, lr=1),
}

def get_trainer(config):
    return TRAINER_DICT[config['trainer']]

def cov_loss_f(pred, target):
    closs = (pred.T.cov() - target.T.cov())**2
    mloss = (pred.mean(0) - target.mean(0))**2
    assert len(mloss) == len(closs)
    return closs.mean()#(mloss.mean() + closs.mean())/2

def mse_loss_f(pred, target, correction_factor=4, weight=1, bias=0, *args, **kwargs):
    return ((pred - target).pow(2) * weight + bias).mean() * correction_factor

class CovLoss(nn.Module):
    def __init__(self, loss=None, alpha=0.5, correction_factor=1):
        super().__init__()
        self.alpha = alpha # percent loss which is covariance loss
        if loss is None:
            self.loss = nn.PoissonNLLLoss(log_input=False, reduction='mean')
        else:
            self.loss = loss
        self.correction_factor = correction_factor
    def forward(self, pred, target, *args, **kwargs):
        return (self.alpha*cov_loss_f(pred, target) + (1-self.alpha)*self.loss(pred, target)) * self.correction_factor

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, *args, **kwargs):
        return mse_loss_f(pred, target)

class SMSE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, *args, **kwargs):
        return mse_loss_f(pred, target, bias=target*(1-pred), correction_factor=2)
    
class SMSE2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, *args, **kwargs):
        inds0 = target==0
        a = 0.5
        l1 = ((pred[inds0] - target[inds0])**2).sum()
        l2 = ((pred[~inds0] - target[~inds0])**2 + (1 - pred[~inds0])**2).sum()
        return (a*l1 + (1-a)*l2) / target.numel() * 3

class SMSE3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, *args, **kwargs):
        inds0 = target==0
        a = 0.5
        b = 0.5
        l1 = ((pred[inds0] - target[inds0])**2).sum()
        l2 = ((pred[~inds0] - target[~inds0])**2 + (1 - pred[~inds0])**2).sum()
        l3 = (1 - 4*((pred - 0.5) ** 2)).sum() / 100
        return (b*(a*l1 + (1-a)*l2) + (1-b)*l3) / target.numel() * 3
    
class ZMSE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, *args, **kwargs):
        inds = target==0
        return mse_loss_f(pred[inds], target[inds], correction_factor=1) + mse_loss_f(pred[~inds], target[~inds], correction_factor=1/10)

class ZMSE2(nn.Module):
    def __init__(self, eps=1e-1):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target, *args, **kwargs):
        w = torch.abs(target) + self.eps
        return mse_loss_f(pred, target, weight=w)
# class NanLoss(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.loss = NDNT.metrics.poisson_loss.PoissonLoss_datafilter(*args, **kwargs)
#     def forward(self, pred, target, *args, **kwargs):
#         out = self.loss(pred, target, *args, **kwargs)
#         if torch.isnan(out):
#             print('nan loss', torch.isnan(pred).sum().item(), torch.isnan(target).sum().item(), torch.isnan(out).sum().item())
#             print(torch.where(torch.isnan(out))[0])
#             print(pred[out.isnan()], target[out.isnan()])
#         return out
    
class ZScoreNL(nn.Module):
    def forward(self, x):
        return torch.exp(torch.abs(x)) * x
class CubeNL(nn.Module):
    def forward(self, x):
        return x**3

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

def robust_loss_grad(t, c):
    t = t.clone()
    inds = torch.abs(t)<c
    t[inds] = torch.sign(t[inds])
    t[~inds] = 2 * t[~inds] / c
    return t
    
class znorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, batch_n, mean, std):
        ctx.save_for_backward(x, torch.tensor([batch_n]), mean, std)
        return (x - mean) / std

    @staticmethod
    def backward(ctx, grad_output):
        x, batch_n, mean, std = ctx.saved_tensors
        stdx, meanx = x.std(0), x.mean(0)        
        dx = grad_output / std
        dm = -grad_output / std
        ds = -grad_output * (x - mean) / std**2
        
        dm1 = -robust_loss_grad(meanx.unsqueeze(0) - mean, 0.01)
        ds1 = -robust_loss_grad(stdx.unsqueeze(0) - std, 0.01)
        dlddz = robust_loss_grad((x - mean) / std - (x - meanx) / stdx, 0.01) 
        dldx = dlddz / std - dlddz / stdx
        
        # w = torch.sigmoid(batch_n/1000 - 10).item()
        # wx = (dldx**2).mean(0) /  (dx**2).mean(0) * 1
        c = 0.1
        wm = (dm**2).mean(0) /  (dm1**2).mean(0) * c
        ws = (ds**2).mean(0) /  (ds1**2).mean(0) * c
        return dx, None, dm*(1-c) + dm1*wm , ds*(1-c) + ds1*ws

class ZNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, dims))
        self.std = nn.Parameter(torch.ones(1, dims))
        self.batch = -1
    def forward(self, x):
        self.batch += 1
        return znorm.apply(x, self.batch, self.mean, self.std)
    
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

class scalegrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(torch.tensor([scale]))
        return x
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        return grad_output*scale.item(), None
        
class LogLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, *args, **kwargs):
        # w = torch.mean(y, 0).repeat(x.shape[0], 1)
        # w = torch.clip(w, 1e-3, 1-1e-3)
        a = 0.9
        inds0 = y == 0
        loss0 = - torch.log(torch.clip(1-x[inds0], 1e-3, 1))
        loss1 = - torch.log(torch.clip(x[~inds0], 1e-3, 1))
        return (loss0.mean()*a+loss1.mean()*(1-a)) / 10

LOSS_DICT = {
    'mse': MSE,
    'zmse': MSE,
    'czmse': ZMSE,
    'czmse2': ZMSE2,
    'poisson': NDNT.metrics.poisson_loss.PoissonLoss_datafilter,
    'cov': CovLoss,
    'log': LogLoss,
    'smse': SMSE,
    'smse2': SMSE2,
    'smse2_cov': lambda: CovLoss(loss=SMSE2(), alpha=0.5, correction_factor=3),
    'smse3': SMSE3,
}

NONLINEARITY_DICT = {
    **{k: nn.Softplus() for k in ['poisson', 'cov']},
    **{k: nn.Sigmoid() for k in ['log', 'smse', 'smse2', 'smse2_cov', 'smse3']},
    **{k: nn.Identity() for k in ['mse', 'zmse', 'czmse', 'czmse2']},
}

def get_loss(config):
    return LOSS_DICT[config['loss']](), NONLINEARITY_DICT[config['loss']]