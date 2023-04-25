import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import einops
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.base import ModelWrapper
    
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

def timeIsHidden(fn, x):
    '''
        Hide time dimension.
    '''
    b, c, t, h, w = x.shape
    x = rearrange(x, "b c t h w -> (b t) c h w", b=b, t=t, c=c, h=h, w=w)
    x = fn(x)
    x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
    return x

def timeIsChannel(fn, x):
    '''
        Use time as channel dimension.
    '''
    b, c, t, h, w = x.shape
    x = x.reshape(b*c, t, h, w) # "b c t h w -> (b c) t h w"
    x = fn(x)
    x = rearrange(x, "(b c) t h w -> b c t h w", b=b, c=c)
    return x

def spaceIsHidden(fn, x):
    '''
        Hide space dimension.
    '''
    b, c, t, h, w = x.shape
    x = rearrange(x, "b c t h w -> (b h w) c t 1", b=b, t=t, c=c, h=h, w=w)
    x = fn(x)
    x = rearrange(x, "(b h w) c t 1 -> b c t h w", b=b, h=h, w=w)
    return x

def spaceIsChannel(fn, x):
    '''
        Use space as channel dimension.
    '''
    b, c, t, h, w = x.shape
    x = rearrange(x, "b c t h w -> (b c) (h w) t 1", b=b, c=c, t=t, h=h, w=w)
    x = fn(x)
    x = rearrange(x, "(b c) hw t 1 -> b c hw t 1", b=b, c=c)
    return x

def foldTime(x):
    b, c, t, h, w = x.shape
    return rearrange(x, "b c t h w -> b (c t) h w", b=b, t=t, c=c, h=h, w=w)

# class AttentionGain(nn.Module):
#     '''
#         Dynamic gain per pixel.
#     '''
#     def __init__(self, input_dims):
#         super().__init__()
#         c, t, h, w = input_dims
#         self.to_qkv_spatial = nn.Conv2d(t, 3, 1, bias=False)
#         self.to_qkv_temporal = nn.Conv2d(h * w, 3, 1, bias=False)

#     def forward(self, x):
#         bypass = x.clone()
#         b, c, t, h, w = x.shape
#         qkv_spatial = timeIsChannel(self.to_qkv_spatial, x)
#         qkv_spatial = qkv_spatial.chunk(3, dim=-3) # each is (b, c, 1, h, w)
#         qkv_temporal = spaceIsChannel(self.to_qkv_temporal, x).chunk(3, dim=-3) # each is (b, c, 1, t, 1)
#         qs, ks, vs = map(
#             lambda t: t.squeeze(2).reshape(b, c, h*w), qkv_spatial
#         )
#         qt, kt, vt = map(
#             lambda t: t.squeeze(2).squeeze(3), qkv_temporal
#         )
        
#         qs = qs.softmax(dim=-2).unsqueeze(-2)
#         ks = ks.softmax(dim=-1).unsqueeze(-2)
#         vs = vs.unsqueeze(-2)
#         qt = qt.softmax(dim=-2).unsqueeze(-1)
#         kt = kt.softmax(dim=-1).unsqueeze(-1)
#         vt = vt.unsqueeze(-1)
#         eps = 1e-10
#         k = torch.log(kt+eps) + torch.log(ks+eps)
#         q = torch.log(qt+eps) + torch.log(qs+eps)
#         v = vt * vs
#         context = torch.exp(k/4 + q/4) * v
#         out = context.reshape(b, c, t, h, w) #rearrange(context, "b c t (x y) -> b c t x y", b=b, c=c, t=t, x=h, y=w)
#         return out + bypass

class ImageLinearAttention(nn.Module):
    def __init__(self, input_dims, key_dim = 2, value_dim = 2, norm_queries = True):
        super().__init__()
        c, t, h, w = input_dims
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.norm_queries = norm_queries
        self.to_qkv_spatial = nn.Sequential(
            # WeightStandardizedConv2d(t, key_dim*3, 1, bias=False),
            nn.Conv2d(t, key_dim*3, 1, bias=False),
            # nn.GroupNorm(3, 6),
        )
        self.to_qkv_temporal = nn.Sequential(
            # WeightStandardizedConv2d(h * w, key_dim*3, 1, bias=False),
            nn.Conv2d(h * w, key_dim*3, 1, bias=False),
            # nn.GroupNorm(3, 6),
        )
        self.to_out_spatial = nn.Conv2d(c*value_dim, c, 1, bias=False)
        self.to_out_temporal = nn.Conv2d(c*value_dim, c, 1, bias=False)
        

    def forward(self, x):
        b, c, t, h, w = x.shape        
        qkv_spatial = timeIsChannel(self.to_qkv_spatial, x).chunk(3, dim=-3) # each is (b, c, 2, h, w)
        qkv_temporal = spaceIsChannel(self.to_qkv_temporal, x).chunk(3, dim=-3) # each is (b, c, 2, t, 1)
        qs, ks, vs = map(
            lambda t: t.reshape(b, c, 2, h*w), qkv_spatial # each is (b, c, 2, h*w)
        )
        qt, kt, vt = map(
            lambda t: t.squeeze(-1), qkv_temporal # each is (b, c, 2, t)
        )
        if self.norm_queries:
            qs = qs.softmax(dim=-2).unsqueeze(-2)
            qt = qt.softmax(dim=-2).unsqueeze(-1)
        ks = ks.softmax(dim=-1).unsqueeze(-2)
        kt = kt.softmax(dim=-1).unsqueeze(-1)
        vs = vs.unsqueeze(-2)
        vt = vt.unsqueeze(-1)
        eps = 1e-10
        k = torch.exp((torch.log(kt+eps) + torch.log(ks+eps))/2) * self.key_dim ** -0.25
        q = torch.exp((torch.log(qt+eps) + torch.log(qs+eps))/2) * self.key_dim ** -0.25
        v = (vt * vs) ** 0.5
        context = torch.einsum('bcdtn,bcetn->bcde', k, v)
        out = torch.einsum('bcdtn,bcde->bcetn', q, context)
        out = out.reshape(b, c, self.value_dim, t, h*w)
        out = foldTime(out) # (b, c*vdim, t, h*w)
        out = out.reshape(b, c*self.value_dim, t, h, w)
        out_s = timeIsHidden(self.to_out_spatial, out)
        out_t = spaceIsHidden(self.to_out_temporal, out)
        out = (out_s * out_t) ** 0.5
        return out

class GainControl(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        c, t, h, w = input_dims
        self.to_kv_spatial = nn.Sequential(
            nn.Conv2d(t, 2, 1, bias=False),
        )
        self.to_kv_temporal = nn.Sequential(
            nn.Conv2d(h * w, 2, 1, bias=False),
        )
        
    def forward(self, x):
        if x.isnan().any():
            print('nan in x')
        b, c, t, h, w = x.shape        
        kv_spatial = timeIsChannel(self.to_kv_spatial, x).chunk(2, dim=-3) # each is (b, c, 2, h, w)
        kv_temporal = spaceIsChannel(self.to_kv_temporal, x).chunk(2, dim=-3) # each is (b, c, 2, t, 1)
        ks, vs = map(
            lambda t: t.reshape(b, c, h*w), kv_spatial # each is (b, c, h*w)
        )
        kt, vt = map(
            lambda t: t.squeeze(-1).squeeze(-2), kv_temporal # each is (b, c, t)
        )
        if ks.isnan().any():
            print('nan in ks')
        if kt.isnan().any():
            print('nan in kt')
            raise KeyboardInterrupt
        if vs.isnan().any():
            print('nan in vs')
        if vt.isnan().any():
            print('nan in vt')
            raise KeyboardInterrupt
        ks = ks.softmax(dim=-1).unsqueeze(-2)
        kt = kt.softmax(dim=-1).unsqueeze(-1)
        vs = vs.unsqueeze(-2)
        vt = vt.unsqueeze(-1)
        k = (kt * ks) ** 0.5
        if k.isnan().any():
            print('nan in k')
        v = vt * vs
        if v.isnan().any():
            print('nan in v')
            raise KeyboardInterrupt
        out = k * v # (b, c, t, h*w)
        if out.isnan().any():
            print('nan in out')
            raise KeyboardInterrupt
        out = out.reshape(b, c, t, h, w)
        return out

class ConeLayer(nn.Module): #TODO should be 6 layers 3 each for inhib and excite
    def __init__(self, input_dims, gain=False):
        super().__init__()
        c, t, h, w = input_dims #[1, 24, 35, 35]
        assert c == 1
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.SiLU()
        ) 
        self.temporal = nn.Sequential(
            nn.Conv2d(3, 3, (7, 1), padding=(3, 0), groups=3),
            nn.SiLU()
        )
        if gain == True:
            self.gain = GainControl([3, t, h, w])
        elif gain == False:
            self.gain = nn.Sequential(nn.SiLU(), SwitchNorm3d(3))
        else:
            self.gain = gain([3, t, h, w])
    
    def forward(self, x):
        x = timeIsHidden(self.spatial, x)
        if x.isnan().any():
            print('nan in space cone')
        x = spaceIsHidden(self.temporal, x)
        if x.isnan().any():
            print('nan in time cone')
            raise KeyboardInterrupt
        return self.gain(x)

class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.zmse_safe = True
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias
    
class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.reshape(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.reshape(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean = self.running_mean + (1 - self.momentum) * mean_bn.data
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias
    
class DotGain(torch.nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims
        weight_shapes = [
            [1] + [input_dims[i] if i==j else 1 for j in range(len(input_dims))] for i in range(len(input_dims))
        ]
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.rand(sz) - 0.5) for sz in weight_shapes]
        )
        self.scale = torch.nn.Parameter(torch.zeros(len(input_dims)))
        self.bias = torch.nn.Parameter(torch.zeros(len(input_dims)))
    def forward(self, x):
        assert list(x.shape[1:]) == self.input_dims, f"Expected input shape {self.input_dims}, got {x.shape[1:]}"
        b = x.shape[0]
        gain = torch.zeros((x.shape[0], 1), device=x.device)
        for i in range(len(self.weights)):
            ein_sum = (x * self.weights[i]).sum(tuple(range(1, len(x.shape)))).unsqueeze(1)
            norm = ((self.weights[i]**2).sum() * (x**2).reshape(b, -1).sum(1))**0.5
            gain_i = self.scale[i]*ein_sum/norm.unsqueeze(1) + self.bias[i]
            gain += gain_i
        gain = torch.exp(gain).view(b, *[1 for _ in x.shape[1:]]) 
        return gain * x

# class BioV(nn.Module):
#     def __init__(self, input_dims, cids, device, gain=True, bypass=False):
#         super().__init__()
#         self.bypass = bypass
#         self.cids = cids
#         self.input_dims = input_dims
#         c, h, w, t = input_dims #[1, 35, 35, 24]
#         input_dims = [c, t, h, w] # adjusted input dims
#         self.conelayer = ConeLayer(input_dims, gain=DotGain if gain else False)
#         self.bplayer = nn.Sequential(
#             nn.Conv2d(t, t//2, 5, padding=2),
#             # nn.GroupNorm(1, t//2),
#             nn.SiLU(),
#         )# time is channel
#         self.bpbypass = nn.Conv2d(t, t//2, 1) if bypass else lambda x: x[:, :t//2, ...]
#         self.retinal_gain = DotGain([t//2, 3, h, w]) if gain else nn.Sequential(nn.SiLU(), nn.BatchNorm3d(t//2))
#         self.lgn = nn.Sequential(
#             nn.Conv2d((t//2)*3, 32, 3, stride=2, padding=1),
#             # nn.GroupNorm(1, 32),
#             nn.SiLU(),
#         )
#         self.lgn_bypass = nn.Conv2d((t//2)*3, 1, 3, stride=2, padding=1) if bypass else lambda x: x[:, :1, ::2, ::2]
#         nh, nw = math.ceil(h/2), math.ceil(w/2)
#         self.lgn_gain = DotGain([32, 1, nh, nw]) if gain else nn.Sequential(nn.SiLU(), nn.BatchNorm3d(32))
#         self.lgn_out = nn.Sequential(
#             nn.Conv2d(32, len(cids), 3, stride=2, padding=1),
#             # nn.GroupNorm(len(cids), len(cids)),
#             nn.SiLU(),
#         )
#         nh, nw = math.ceil(nh/2), math.ceil(nw/2)
#         self.adapter_gain = nn.ModuleList([
#             DotGain([1, 1, nh, nw]) if gain else nn.Sequential(nn.SiLU(), SwitchNorm3d(1)) for _ in range(len(cids))
#         ])
#         self.out_layers = nn.ModuleList([
#             nn.Linear(nh*nw, 1) for _ in range(len(cids))
#         ])
#         self.output_NL = nn.Softplus() #nn.ReLU()

#     def forward(self, x):
#         toBypass = self.bypass
#         b, c, h, w, t = x.shape[0], *self.input_dims
#         if x.ndim == 2:
#             x = x.reshape(b, c, h, w, t)
#         x = rearrange(x, "b c h w t -> b c t h w", b=b, c=c, h=h, w=w, t=t)
        
#         x = self.conelayer(x)
#         if x.isnan().any():
#             print("cone gain nan")
#         bypass = timeIsChannel(self.bpbypass, x) if toBypass else 0
#         x = timeIsChannel(self.bplayer, x)
#         x = rearrange(x, "b c t h w -> b t c h w", b=b, c=3, t=t//2, h=h, w=w)
#         bypass = rearrange(bypass, "b c t h w -> b t c h w", b=b, c=3, t=t//2, h=h, w=w) if toBypass else 0
#         x = self.retinal_gain(x) + bypass
        
#         if x.isnan().any():
#             print("retina gain nan")
        
#         x = foldTime(x) # back to b c h w
#         bypass = self.lgn_bypass(x) if toBypass else 0
#         x = self.lgn(x) # b 32 h//2 w//2
#         x = self.lgn_gain(x.unsqueeze(2)).squeeze(2) # b 32 h//2 w//2
#         x += bypass
#         x = self.lgn_out(x) # b cids h//4 w//4
#         if x.isnan().any():
#             print("lgn_out nan")
        
#         out = torch.empty((b, len(self.out_layers)), device=x.device)
#         for i in range(len(self.out_layers)):
#             bypass = x[:, i].unsqueeze(1).unsqueeze(1) if toBypass else 0
#             temp = self.adapter_gain[i](x[:, i].unsqueeze(1).unsqueeze(1)) + bypass
            
#             out[:, i] = self.out_layers[i](temp.flatten(1)).squeeze(1)
#         # return torch.exp(out)
#         return self.output_NL(out.reshape(b, len(self.out_layers)))
    
class BioV(nn.Module):
    def __init__(self, input_dims, cids, device, gain=False, bypass=False):
        super().__init__()
        self.bypass = bypass
        self.cids = cids
        self.input_dims = input_dims
        c, h, w, t = input_dims #[1, 35, 35, 24]
        input_dims = [c, t, h, w] # adjusted input dims
        self.conelayer = ConeLayer(input_dims, gain=gain)
        self.bplayer = nn.Sequential(
            nn.Conv2d(t, t//2, 5, padding=2),
            # nn.GroupNorm(1, t//2),
            nn.SiLU(),
        )# time is channel
        self.bpbypass = nn.Conv2d(t, t//2, 1) if bypass else lambda x: x[:, :t//2, ...]
        self.retinal_gain = GainControl([t//2, 3, h, w]) if gain else nn.Sequential(nn.SiLU(), nn.BatchNorm3d(t//2))
        self.lgn = nn.Sequential(
            nn.Conv2d((t//2)*3, 32, 3, stride=2, padding=1),
            # nn.GroupNorm(1, 32),
            nn.SiLU(),
        )
        self.lgn_bypass = nn.Conv2d((t//2)*3, 1, 3, stride=2, padding=1) if bypass else lambda x: x[:, :1, ::2, ::2]
        nh, nw = math.ceil(h/2), math.ceil(w/2)
        self.lgn_gain = GainControl([32, 1, nh, nw]) if gain else nn.Sequential(nn.SiLU(), nn.BatchNorm3d(32))
        self.lgn_out = nn.Sequential(
            nn.Conv2d(32, len(cids), 3, stride=2, padding=1),
            # nn.GroupNorm(len(cids), len(cids)),
            nn.SiLU(),
        )
        nh, nw = math.ceil(nh/2), math.ceil(nw/2)
        self.adapter_gain = nn.ModuleList([
            GainControl([1, 1, nh, nw]) if gain else nn.Sequential(nn.SiLU(), SwitchNorm3d(1)) for _ in range(len(cids))
        ])
        self.out_layers = nn.ModuleList([
            nn.Linear(nh*nw, 1) for _ in range(len(cids))
        ])
        self.output_NL = SwitchNorm1d(len(cids)) #nn.Softplus() #nn.ReLU()

    def forward(self, x):
        toBypass = self.bypass
        b, c, h, w, t = x.shape[0], *self.input_dims
        if x.ndim == 2:
            x = x.reshape(b, c, h, w, t)
        x = rearrange(x, "b c h w t -> b c t h w", b=b, c=c, h=h, w=w, t=t)
        
        x = self.conelayer(x)
        if x.isnan().any():
            print("cone gain nan")
        bypass = timeIsChannel(self.bpbypass, x) if toBypass else 0
        x = timeIsChannel(self.bplayer, x)
        x = rearrange(x, "b c t h w -> b t c h w", b=b, c=3, t=t//2, h=h, w=w)
        bypass = rearrange(bypass, "b c t h w -> b t c h w", b=b, c=3, t=t//2, h=h, w=w) if toBypass else 0
        x = self.retinal_gain(x) + bypass
        
        if x.isnan().any():
            print("retina gain nan")
        
        x = foldTime(x) # back to b c h w
        bypass = self.lgn_bypass(x) if toBypass else 0
        x = self.lgn(x) # b 32 h//2 w//2
        x = self.lgn_gain(x.unsqueeze(2)).squeeze(2) # b 32 h//2 w//2
        x += bypass
        x = self.lgn_out(x) # b cids h//4 w//4
        if x.isnan().any():
            print("lgn_out nan")
        
        out = torch.empty((b, len(self.out_layers)), device=x.device)
        for i in range(len(self.out_layers)):
            bypass = x[:, i].unsqueeze(1).unsqueeze(1) if toBypass else 0
            temp = self.adapter_gain[i](x[:, i].unsqueeze(1).unsqueeze(1)) + bypass
            
            out[:, i] = self.out_layers[i](temp.flatten(1)).squeeze(1)
        # return torch.exp(out)
        return self.output_NL(out.reshape(b, len(self.out_layers)))

class Block(nn.Module):
    def __init__(self, dim, dim_out, k=3, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, k, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class LinearAttentionReadoutFunction(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        '''
            dim is len of the input vector
            heads is the number of heads (channels of channels)
        '''
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
        #                             nn.GroupNorm(1, dim))
        self.to_out = nn.GroupNorm(1, self.heads)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = self.to_out(out)
        out = rearrange(out, "b h c (x y) -> b h c x y", h=self.heads, x=h, y=w)
        return out
        # return self.to_out(out)
        
class PytorchWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_reg_loss(self, *args, **kwargs):
        return 0
    def prepare_regularization(self, normalize_reg=False):
        return 0
    def forward(self, x, pass_dict=False, *args, **kwargs):
        if type(x) is dict and not pass_dict:
            x = x['stim']
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        return self.model(x, *args, **kwargs)

class LinearAttentionReadout():
    def __init__(self, in_dim, cids, head_dim=2):
        '''
            in_dim is input dimension
            heads is the number of heads (channels of channels)
        '''
        super().__init__()
        c, y, x, _ = in_dim
        self.in_dims = in_dim
        self.cids = cids
        self.head_dim = head_dim
        self.attention = PreNorm(
            c,
            LinearAttentionReadoutFunction(
                c,
                len(cids),
                head_dim
            )
        )
        self.reducer = nn.Conv2d(c, head_dim, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(x * y * head_dim, c),
            nn.GELU(),
        )
        self.mlps = nn.ModuleList([
            nn.Linear(c, 1) for _ in cids
        ])
    
    def forward(self, x):
        x = x.squeeze(-1)
        b, c, w, h = x.shape
        reduced = self.reducer(x).reshape(b, 1, -1)
        x = self.attention(x).reshape(b, len(self.cids), -1) # b h (c x y)
        x = x + reduced
        x = self.shared_mlp(x.reshape(b*len(self.cids), -1))
        x = rearrange(x, "(b h) c -> b h c", h=len(self.cids), b=b)
        return torch.cat([mlp(x[:, i]) for i, mlp in enumerate(self.mlps)], dim=1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(dim),
        #     nn.Linear(dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim),
        # )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in),
                        block_klass(dim_in, dim_in),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out),
                        block_klass(dim_out + dim_in, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x)
        return self.final_conv(x)

from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

class UNet(nn.Module):
    def __init__(self, cids):
        super().__init__()
        self.cids = cids
        self.core = nn.Sequential(
            nn.Upsample(36, mode="bilinear"),
            Unet(
                dim=36,
                channels=24,
                dim_mults=(1, 2, 4,),
                # init_dim=12,
                out_dim=1,
            )
        )
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36*36, len(cids)),
            nn.Softplus(),
        )
        
    def compute_reg_loss(self):
        return 0

    def prepare_regularization(self, normalize_reg = False):
        return 0

    def forward(self, x):
        if isinstance(x, dict):
            x = x["stim"]
        if x.ndim == 2:
            x = x.reshape(-1, 35, 35, 24)
        x = x.permute(0, 3, 1, 2)
        x = self.core(x)
        return self.readout(x)
    