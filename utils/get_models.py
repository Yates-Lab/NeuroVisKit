from .utils import initialize_gaussian_envelope, seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import ModelWrapper, CNNdense
# from unet import UNet, BioV
import math
from tqdm import tqdm
import moten

class PytorchWrapper(ModelWrapper):
    def __init__(self, *args, cids=None, bypass_preprocess=False, **kwargs):
        super().__init__(*args, cids=cids, **kwargs)
        self.bypass_preprocess = bypass_preprocess
        if not hasattr(self, 'cids'):
            self.cids = cids
    def compute_reg_loss(self, *args, **kwargs):
        loss = 0
        if hasattr(self.model, 'compute_reg_loss'):
            loss = self.model.compute_reg_loss(*args, **kwargs)
        if type(loss) is int:
            return torch.tensor(loss, device=self.parameters().__next__().device).float()
        return loss
    def prepare_regularization(self, normalize_reg=False):
        if hasattr(self.model, 'prepare_regularization'):
            return self.model.prepare_regularization(normalize_reg=normalize_reg)
        return 0
    def forward(self, x, pass_dict=False, *args, **kwargs):
        if not self.bypass_preprocess:
            if type(x) is dict and not pass_dict:
                x = x['stim']
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim == 4:
                x = x.unsqueeze(1)
        return self.model(x, *args, **kwargs)
    
# def get_biov(config_init):
#     def get_biov_helper(config, device='cpu'):
#         seed_everything(config_init['seed'])
#         cids = config['cids']
#         input_dims = config_init['input_dims']
#         device = config_init['device']
#         pmodel = BioV(input_dims, cids, device)
#         if 'lightning' not in config_init or not config_init['lightning']:
#             pmodel.to(device)
#         return PytorchWrapper(pmodel)
#     return get_biov_helper

# def get_unet(config_init):
#     def get_unet_helper(config, device='cpu'):
#         seed_everything(config_init['seed'])
#         cids = config['cids']
#         model = ModelWrapper(UNet(cids))
#         if 'lightning' not in config_init or not config_init['lightning']:
#             model.to(device)
#         return model
#     return get_unet_helper

# class gaborC(nn.Module):
#     #assumes input has been preprocessed with gabors
#     def __init__(self, cids, nl=nn.Softplus()):
#         super().__init__()
#         self.nfilters = moten.get_default_pyramid(vhsize=(70, 70), fps=240).nfilters
#         self.cids = cids
#         self.conv = nn.Conv1d(self.nfilters*len(cids), len(cids), kernel_size=36, groups=len(cids))
#         self.filter_gain = nn.Parameter(torch.randn(len(cids), self.nfilters, 1))
#         self.filter_bias = nn.Parameter(torch.randn(len(cids), self.nfilters, 1))
#         self.output_NL = nl
#     def forward(self, x):
#         assert len(x["stim"]) == len(x["robs"])+35, "stim and robs must have same length: stim: {}, robs: {}".format(len(x["stim"]), len(x["robs"]))
#         x = F.pad(x["stim"].flatten(1).T, (35, 0))
#         x = x.unsqueeze(0) * self.filter_gain + self.filter_bias # shaped cids, nfilters, time
#         x = F.softmax(x, 0)
#         x = x.reshape(1, len(self.cids)*self.nfilters, -1)
#         return self.output_NL(self.conv(x).squeeze(0).T[35:])
#     def compute_reg_loss(self, *args, **kwargs):
#         return torch.abs(self.conv.weight).mean()/10

def gain(x, dim):
    alpha = 1
    x_sub = 1 + (torch.abs(x)**alpha).sum(dim=dim, keepdim=True)
    return x / x_sub

class self_attention1d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        # self.bn = nn.BatchNorm1d(out_dim)
        self.reg = 0
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(-1, self.out_dim, 1)
        k = k.reshape(-1, self.out_dim, 1)
        v = v.reshape(-1, self.out_dim, 1)
        x = torch.bmm(q, k.transpose(1, 2))
        # x = F.softmax(x, 2)
        # x = self.bn(x)
        x = gain(x, 2)
        self.reg = torch.abs(x).mean()
        x = torch.bmm(x, v)
        return x.reshape(-1, self.out_dim)

# class gaborC(nn.Module):
#     #assumes input has been preprocessed with gabors
#     def __init__(self, cids, nl=nn.Softplus()):
#         super().__init__()
#         self.nfilters = moten.get_default_pyramid(vhsize=(70, 70), fps=240).nfilters
#         self.cids = cids
#         self.atten = self_attention1d(self.nfilters, self.nfilters//4)
#         self.conv = nn.Conv1d(self.nfilters//4, len(cids), kernel_size=36)
#         self.output_NL = nl
#     def forward(self, x):
#         assert len(x["stim"]) == len(x["robs"])+35, "stim and robs must have same length: stim: {}, robs: {}".format(len(x["stim"]), len(x["robs"]))
#         x = self.atten(x["stim"])
#         x = F.pad(x.T, (35, 0)).unsqueeze(0)
#         return self.output_NL(self.conv(x).squeeze(0).T[35:])
#     def compute_reg_loss(self, *args, **kwargs):
#         return self.atten.reg #torch.abs(self.conv.weight).mean()/100 #+ torch.abs(self.mlp[0].weight).mean()/100 + torch.abs(self.mlp[2].weight).mean()/100
def dl_to_data(dl):
    return {k: torch.cat([d[k] for d in dl]) for k in next(iter(dl)).keys()}

class gaborC(nn.Module):
    #assumes input has been preprocessed with gabors
    def __init__(self, cids, nl=nn.Softplus()):
        super().__init__()
        self.nfilters = 28084#moten.get_default_pyramid(vhsize=(70, 70), fps=45).nfilters
        self.cids = cids
        self.linear = nn.Linear(self.nfilters, len(cids))
        # self.linear.weight.data = torch.zeros_like(self.linear.weight.data)
        # self.linear.bias.data = torch.zeros_like(self.linear.bias.data)
        self.lag = 0
        self.output_NL = nl
    def forward(self, x):
        return self.output_NL(self.linear(x["stim"]))
    def compute_reg_loss(self, *args, **kwargs):
        return torch.abs(self.linear.weight).mean()*10
    def prepare_model(self, train_dl, *args, **kwargs):
        if True:
            data = dl_to_data(train_dl)
            nfilters = data["stim"].shape[1]
            stim, robs, dfs = data["stim"], data["robs"], data["dfs"]
            stas = torch.zeros((65, nfilters), device=stim.device)
            n = torch.zeros((65), device=stim.device)
            # stim = (stim-stim.mean(0))/stim.std(0)
            for i in range(len(robs)):
                weights = (robs[i]*dfs[i]).reshape(-1, 1)
                corr = stim[i].unsqueeze(0)
                stas += corr*weights
                n+=dfs[i]
            self.linear.weight.data[:, :] = (stas / n.reshape(65, 1))[self.cids, :].to(stim.device)/100
    
def compute_l1_mlp_loss(mlp):
    loss = 0
    num_layers = 0
    for layer in mlp:
        if hasattr(layer, 'weight'):
            loss += torch.abs(layer.weight).mean()
            num_layers += 1
    return loss/num_layers
            
def get_gaborC(config_init):
    def get_gaborC_helper(config, device='cpu'):
        seed_everything(config_init['seed'])
        model = PytorchWrapper(gaborC(config['cids']), bypass_preprocess=True)
        if 'lightning' not in config_init or not config_init['lightning']:
            model.to(device)
        return model
    return get_gaborC_helper

def pad_causal(x, layer, kdims=None):
    kdims = kdims if kdims is not None else np.arange(len(layer.kernel_size))
    pad_array = []
    for i in np.arange(len(layer.kernel_size))[::-1]:
        pad_array += [layer.kernel_size[i]-1, 0] if i in kdims else [0, 0]
    return F.pad(x, pad_array)

class separable_conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super().__init__()
        k = (k, k, k) if type(k) is int else k # x, y, t
        #input shape will be (1, in_channels, x, y, t)
        self.space_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(k[0], k[1], 1), padding="same", groups=math.gcd(in_channels, out_channels))
        self.time_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, k[2]), groups=out_channels)
    def forward(self, x):
        # x shape is (1, in_channels, x, y, t)
        x = self.space_conv(x)
        x = self.time_conv(pad_causal(x, self.time_conv))
        return x

class split_nonlinearity(nn.Module):
    def __init__(self, f=nn.Softplus()):
        super().__init__()
        self.f = f
    def forward(self, x):
        return torch.cat([self.f(x), self.f(-x)], dim=1)
    
# class separable_res_block(nn.Module):
#     def __init__(self, in_channels, out_channels, k=3, nl=nn.Softplus()):
#         super().__init__()
#         assert out_channels % 2 == 0, "out_channels must be even"
#         self.bn = nn.BatchNorm3d(in_channels)
#         self.conv = separable_conv_layer(in_channels, out_channels, k)
#         self.split_nl = nl #split_nonlinearity(nl)
#         self.resize = nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=math.gcd(in_channels, out_channels))
#     def forward(self, x):
#         x = self.bn(x)
#         y = self.split_nl(self.conv(x))
#         return y + self.resize(x)
    
def oddify(i):
    return i - 1 + i % 2
def locality(x):
    # calculates the locality of the energy of a batch of images in shape b, nx, ny
    tx, ty = torch.meshgrid(torch.linspace(-1, 1, oddify(x.shape[1])), torch.linspace(-1, 1, oddify(x.shape[2])), indexing="ij")
    locality_kernel = torch.sqrt(tx**2 + ty**2).to(x.device).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x.unsqueeze(1)**2, locality_kernel, padding="valid").mean()
def locality_loss(layer):
    if 1 in layer.kernel_size:
        squeeze_dim =  1+list(layer.kernel_size).index(1)
        filt = layer.weight.reshape(-1, *layer.kernel_size).squeeze(squeeze_dim)
    elif len(layer.kernel_size) == 3:
        filt = layer.weight.permute(0, 1, 4, 2, 3).reshape(-1, *layer.kernel_size[:2])
    return locality(filt)
# class separable_readout(nn.Module):
#     def __init__(self, input_dims, ncids):
#         super().__init__()
#         self.input_dims = input_dims # (c, x, y, t)
#         self.bn = nn.BatchNorm3d(input_dims[0])
#         self.collapse_time = nn.Conv3d(ncids, ncids, kernel_size=(1, 1, input_dims[-1]), groups=ncids)
#         self.collapse_channels = nn.Parameter(torch.rand(ncids, input_dims[0]))
#         self.collapse_space = nn.Parameter(torch.rand(ncids, input_dims[1], input_dims[2]))
#         self.gain = nn.Parameter(torch.ones(1, ncids))
#         self.bias = nn.Parameter(torch.zeros(1, ncids))
#     def forward(self, x):
#         x = self.bn(x)
#         x = torch.einsum("bcxyt,nc->bnxyt", x, self.collapse_channels)
#         x = F.tanh(x)
#         x = torch.einsum("bnxyt,nxy->bnt", x, self.collapse_space).unsqueeze(2).unsqueeze(2)
#         x = F.tanh(x)
#         x = self.collapse_time(pad_causal(x, self.collapse_time))
#         return x.reshape(x.shape[1], x.shape[-1]).T[self.input_dims[-1]-1:] * self.gain + self.bias
#     def compute_reg_loss(self, *args, **kwargs):
#         return (locality_loss(self.collapse_time) + locality(self.collapse_space))/200000

class separable_readout(nn.Module):
    def __init__(self, input_dims, ncids):
        super().__init__()
        self.input_dims = input_dims # (c, x, y, t)
        # self.collapse_channels = nn.Parameter(torch.randn(ncids, input_dims[0]))
        # self.collapse_space = nn.Parameter(torch.randn(ncids, input_dims[1], input_dims[2]))
        self.collaps = nn.Parameter(torch.randn(ncids, input_dims[0], input_dims[1], input_dims[2]))
        self.gain = nn.Parameter(torch.ones(1, ncids))
        self.bias = nn.Parameter(torch.zeros(1, ncids))
    def forward(self, x):
        # x = torch.einsum("bcxy,nxy->bcn", x, self.collapse_space)
        # x = torch.einsum("bcn,nc->bn", x, self.collapse_channels)
        x = torch.einsum("bcxy,ncxy->bn", x, self.collapse)
        return x * self.gain + self.bias
    def compute_reg_loss(self, *args, **kwargs):
        return locality(self.collapse_space)/100000
    
# class dense_readout(nn.Module):
#     def __init__(self, input_dims, ncids):
#         super().__init__()
#         self.input_dims = input_dims # (c, x, y, t)
#         self.bn = nn.BatchNorm3d(input_dims[0])
#         self.readout = nn.Conv3d(input_dims[0], ncids, kernel_size=input_dims[-3:])
#     def forward(self, x):
#         x = self.bn(x)
#         x = pad_causal(x, self.readout, kdims=(2,))
#         x = self.readout(x)
#         return x.reshape(x.shape[1], x.shape[-1]).T[self.input_dims[-1]-1:]
#     def compute_reg_loss(self, *args, **kwargs):
#         return locality_loss(self.readout)/10

class SeparableCore(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims # (1, x, y, nlags)
        self.c1 = nn.Conv3d(input_dims[0], 20, kernel_size=(11, 11, self.input_dims[-1]))
        self.bn1 = nn.BatchNorm2d(20)
        self.c2 = nn.Conv2d(20, 20, kernel_size=11, padding="same")
        self.bn2 = nn.BatchNorm2d(20)
        self.c3 = nn.Conv2d(20, 20, kernel_size=11, padding="same")
        self.bn3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 20, kernel_size=11, padding="same")
        self.bn4 = nn.BatchNorm2d(20)
    def forward(self, x):
        x = F.pad(x, (0, 0, 5, 5, 5, 5))
        x = self.c1(x)
        x = x.squeeze(0).permute(3, 0, 1, 2)
        x = self.bn1(x)
        x = F.softplus(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.softplus(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.softplus(x)
        x = self.c4(x)
        x = self.bn4(x)
        x = F.softplus(x)
        return x
class SeparableCNN(nn.Module):
    def __init__(self, input_dims, ncids, nl=nn.Softplus()):
        super().__init__()
        self.input_dims = input_dims # (1, x, y, nlags)
        self.core = SeparableCore(input_dims)
        self.readout = separable_readout((20, *input_dims[1:]), ncids)
        self.output_NL = nl
    def forward(self, x):
        x = x["stim"].permute(1, 2, 3, 0).unsqueeze(0)
        x = self.core(x)
        return self.output_NL(self.readout(x))
    def compute_reg_loss(self, *args, **kwargs):
        return 0.0#self.readout.compute_reg_loss(*args, **kwargs)/100

def get_separable_cnn(config_init):
    def get_separable_cnn_helper(config, device='cpu'):
        seed_everything(config['seed'])
        model = PytorchWrapper(
            SeparableCNN(config_init['input_dims'], len(config['cids'])),
            bypass_preprocess=True,
            cids=config['cids'],
        )
        # torch.compile(model.model)
        if 'lightning' not in config_init or not config_init['lightning']:
            model.to(device)
        return model
    return get_separable_cnn_helper

def shrink_conv_weights(x,d=2):
    """
        x is a tensor with shape (cout, cin, k1, ..., kd)
        d is the number of dimensions of conv (either 2 or 3)
    """
    for cout in range(x.shape[0]):
        for cin in range(x.shape[1]):
            cuttoff = x[cout, cin].max()*0.5
            inds = torch.where(x[cout, cin] < cuttoff)
            if d == 2:
                sign = torch.sign(x[cout, cin, inds[0], inds[1]])
                mag = F.relu(torch.abs(x[cout, cin, inds[0], inds[1]])-cuttoff/100)
                x[cout, cin, inds[0], inds[1]] = sign * mag
            elif d == 3:
                sign = torch.sign(x[cout, cin, inds[0], inds[1], inds[2]])
                mag = F.relu(torch.abs(x[cout, cin, inds[0], inds[1], inds[2]])-cuttoff/100)
                x[cout, cin, inds[0], inds[1], inds[2]] = sign * mag
    return x
    
class CBlock3D(nn.Module):
    def __init__(self, cin, cout, k=3, padding="same", nl=F.softplus, window=True):
        super().__init__()
        self.c = nn.Conv3d(cin, cout, kernel_size=k, padding=padding)
        self.is_window = window
        if window:
            self.window_on = False
            self.original_weights = None
        self.bn = nn.BatchNorm3d(cout)
        self.nl = nl
    def forward(self, x):
        with torch.no_grad():
            self.c.weight.data = shrink_conv_weights(self.c.weight.data, d=3)
        self.window()
        x = self.c(x)
        x = self.bn(x)
        x = self.nl(x)
        self.unwindow()
        return x
    def eval(self):
        return self.train(False)
    def train(self, mode=True):
        if mode:
            if self.is_window and self.window_on:
                self.unwindow()
        else:
            if self.is_window and not self.window_on:
                self.window()
        return super().train(mode)
    def window(self):
        if not self.is_window:
            return
        with torch.no_grad():
            device = self.c.weight.data.device
            w1, w2, w3 = [torch.hamming_window(self.c.weight.shape[i], periodic=False, device=device) for i in [-3, -2, -1]]
            self.original_weights = self.c.weight.data.clone()
            self.c.weight.data = torch.einsum("i,j,k,noijk->noijk", w1, w2, w3, self.c.weight.data)
            self.window_on = True
    def unwindow(self):
        if not self.is_window:
            return
        with torch.no_grad():
            self.c.weight.data = self.original_weights.to(self.c.weight.data.device)
            self.original_weights = None
            self.window_on = False
            
class CBlock2D(nn.Module):
    def __init__(self, cin, cout, k=3, padding="same", nl=F.softplus, window=True):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, kernel_size=k, padding=padding)
        self.is_window = window
        if window:
            self.window_on = False
            self.original_weights = None
        self.bn = nn.BatchNorm2d(cout)
        self.nl = nl
    def forward(self, x):
        with torch.no_grad():
            self.c.weight.data = shrink_conv_weights(self.c.weight.data, d=2)
        self.window()
        x = self.c(x)
        x = self.bn(x)
        x = self.nl(x)
        self.unwindow()
        return x
    def eval(self):
        return self.train(False)
    def train(self, mode=True):
        if mode:
            if self.is_window and self.window_on:
                self.unwindow()
        else:
            if self.is_window and not self.window_on:
                self.window()
        return super().train(mode)
    def window(self):
        if not self.is_window or self.window_on:
            return
        with torch.no_grad():
            device = self.c.weight.data.device
            w1, w2 = [torch.hamming_window(self.c.weight.shape[i], periodic=False, device=device) for i in [-2, -1]]
            self.original_weights = self.c.weight.data.clone()
            self.c.weight.data = torch.einsum("i,j,noij->noij", w1, w2, self.c.weight.data)
            self.window_on = True
    def unwindow(self):
        if not self.is_window or not self.window_on:
            return
        with torch.no_grad():
            self.c.weight.data = self.original_weights.to(self.c.weight.data.device)
            self.original_weights = None
            self.window_on = False
        
    
class CoreC(nn.Module):
    def __init__(self, input_dims, nl=F.softplus, num_lags=36):
        super().__init__()
        self.input_dims = input_dims
        self.nl = nl
        self.num_lags = num_lags
        self.c1 = CBlock3D(input_dims[0], 20, k=(36, 11, 11), padding=0)
        self.c2 = nn.Sequential(
            CBlock2D(20, 20, k=11, padding="same"),
            CBlock2D(20, 20, k=11, padding="same"),
            CBlock2D(20, 20, k=11, padding="same"),
        )
        # input is (t, *input_dims)
    def forward(self, x):
        # x shape is (t, 1, h, w)
        x = x.squeeze(1).permute(1, 0, 2, 3).unsqueeze(0)
        # x shape is (1, 1, t, h, w)
        x = F.pad(x, (5, 5, 5, 5, self.num_lags-1, 0))
        x = self.c1(x)
        # x shape is (1, 20, t, h, w)
        x = x.squeeze(0).permute(1, 0, 2, 3)
        return self.c2(x) # shape is (t, 20, h, w)

class ReadoutC(nn.Module):
    def __init__(self, input_dims, ncids, nlags=36):
        super().__init__()
        self.input_dims = input_dims # (c, x, y)
        self.nlags = nlags
        self.collapse_channels = nn.Conv3d(input_dims[0], ncids, kernel_size=1)
        self.collaps_time = nn.Conv3d(ncids, ncids, kernel_size=(nlags, 1, 1), groups=ncids, bias=False)
        self.collaps_time.weight.data = torch.ones_like(self.collaps_time.weight.data)/1000
        # self.collapse_channels = nn.Parameter(torch.randn(ncids, input_dims[1]))
        self.collapse_space = nn.Parameter(torch.ones(ncids, input_dims[1], input_dims[2])/1000)
        # self.collapse_time = nn.Parameter(torch.randn(ncids, input_dims[0]))
        # self.collaps = nn.Parameter(torch.randn(ncids, input_dims[0], input_dims[1], input_dims[2]))
        self.gain = nn.Parameter(torch.ones(1, ncids))
        self.bias = nn.Parameter(torch.zeros(1, ncids))
    def forward(self, x):
        # with torch.no_grad():
        #     self.collaps_time.weight.data = F.relu(self.collaps_time.weight.data)
        #     self.collapse_space.data = F.relu(self.collapse_space.data)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)
        x = F.pad(x, (0, 0, 0, 0, 0, self.nlags-1))
        x = self.collaps_time(self.collapse_channels(x)).squeeze(0) # (ncids, t, x, y)
        x = torch.einsum("nbxy,nxy->bn", x, self.collapse_space)
        # x = torch.einsum("bcn,nc->bn", x, self.collapse_channels)
        # x = torch.einsum("bcxy,ncxy->bn", x, self.collapse)
        return x * self.gain + self.bias
    def compute_reg_loss(self, *args, **kwargs):
        return 0#self.collapse_space.norm(1)/self.collapse_space.numel()/1000 #locality(self.collapse_space)/100000
    
class CNNC(nn.Module):
    def __init__(self, input_dims, ncids, nl=F.softplus, output_nl=F.softplus):
        super().__init__()
        # input_dims is (c, x, y, nlags)
        self.core = CoreC(input_dims[:3], nl, input_dims[-1])
        self.readout = ReadoutC([20, *input_dims[1:3]], ncids, input_dims[-1])
        self.output_nl = output_nl
    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        return self.output_nl(x)
    def compute_reg_loss(self, *args, **kwargs):
        reg = 0
        for i in [self.core, self.readout]:
            if hasattr(i, 'compute_reg_loss'):
                reg += i.compute_reg_loss(*args, **kwargs)
        return reg

def get_cnnc(config_init):
    def get_cnnc_helper(config, device='cpu'):
        return PytorchWrapper(
            CNNC(config_init['input_dims'], len(config['cids'])),
            cids=config['cids']
        )
    return get_cnnc_helper
        
# def get_attention_cnn(config_init):
#     def get_attention_cnn_helper(config, device='cpu'):
#         cnn = get_cnn(config_init)(config, device)
#         cnn_out_dims = [cnn.model.core_subunits, *cnn.model.core[-1].output_dims[1:3], 1]
#         new_readout = LinearAttentionReadout(cnn_out_dims, config['cids'])
#         new_readout.to(device)
#         cnn.model.readout = new_readout         
#         return cnn
#     return get_attention_cnn_helper

# class PytorchWrapper(ModelWrapper):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#     def compute_reg_loss(self, *args, **kwargs):
#         return 0
#     def prepare_regularization(self, normalize_reg=False):
#         return 0
#     def forward(self, x, pass_dict=False, *args, **kwargs):
#         if type(x) is dict and not pass_dict:
#             x = x['stim']
#         if x.ndim == 3:
#             x = x.unsqueeze(1)
#         if x.ndim == 4:
#             x = x.unsqueeze(1)
#         return self.model(x, *args, **kwargs)

def get_cnn(config_init):
    def get_cnn_helper(config, device='cpu'):
        '''
            Create new model.
        '''
        seed_everything(config_init['seed'])
        num_layers = config['num_layers']
        num_filters = [config[f'num_filters{i}'] for i in range(num_layers)]
        filter_width = [config[f'filter_width{i}'] for i in range(num_layers)]
        # d2x = config['d2x']
        # d2t = config['d2t']
        # center = config['center']
        # edge_t = config['edge_t']
        scaffold = [len(num_filters)-1]
        num_inh = [0]*len(num_filters)
        if 'device' in config:
            device = config['device']
        # modifiers = {
        #     'stimlist': ['frame_tent', 'fixation_onset'],
        #     'gain': [deepcopy(drift_layer), deepcopy(sac_layer)],
        #     'offset': [deepcopy(drift_layer), deepcopy(sac_layer)],
        #     'stage': 'readout',
        # }

        scaffold = [len(num_filters)-1]
        num_inh = [0]*len(num_filters)
        modifiers = None
        
        input_dims = config_init['input_dims']
        cids = config_init['cids']
        mu = config_init['mu']
        
        cr0 = CNNdense(
                input_dims,
                num_filters,
                filter_width,
                scaffold,
                num_inh,
                is_temporal=False,
                NLtype='softplus',
                batch_norm=True,
                norm_type=0,
                noise_sigma=0,
                NC=len(cids),
                bias=False,
                reg_core=None,
                reg_hidden=None,
                reg_readout={'glocalx':.1, 'l2':0.1},
                reg_vals_feat={'l1':0.01},
                cids=cids,
                modifiers=modifiers,
                window='hamming',
                device=device)

        # initialize parameters
        # for i in range(1):
        #     w_centered = initialize_gaussian_envelope( cr0.core[i].get_weights(to_reshape=False), cr0.core[i].filter_dims)
        #     cr0.core[i].weight.data = torch.tensor(w_centered, dtype=torch.float32)
        # if mu is not None and hasattr(cr0.readout, 'mu'):
        #     print('setting mu')
        #     cr0.readout.mu.data = torch.from_numpy(mu[cids].copy().astype('float32')).to(device)
        #     cr0.readout.mu.requires_grad = True
        #     cr0.readout.sigma.data.fill_(0.5)
        #     cr0.readout.sigma.requires_grad = True
        # else:
        #     print('not setting mu', mu is not None, hasattr(cr0.readout, 'mu'))
        model = ModelWrapper(cr0)
        if 'lightning' not in config_init or not config_init['lightning']:
            model.to(device)
        model.prepare_regularization()
        return model
    return get_cnn_helper

MODEL_DICT = {
    'CNNdense': get_cnn,
    # 'UNet': get_unet,
    # 'AttentionCNN': get_attention_cnn,
    # 'BioV': get_biov,
    'gaborC': get_gaborC,
    'seperableCNN': get_separable_cnn,
    'cnnc': get_cnnc,
}

def verify_config(config):
    '''
        Convert config to the format supported by Ray.
    '''
    if 'filters' in config and 'kernels' in config:
        config.update({
            **{f'num_filters{i}': config['filters'][i] for i in range(len(config['filters']))},
            **{f'filter_width{i}': config['kernels'][i] for i in range(len(config['kernels']))},
            'num_layers': len(config['filters']),
        })
    
def get_model(config, factory=False):
    verify_config(config)
    if not factory:
        return MODEL_DICT[config['model']](config)(config)
    return MODEL_DICT[config['model']](config)

# def get_biov(config_init):
#     def get_biov_helper(config, device='cpu'):
#         seed_everything(config_init['seed'])
#         cids = config['cids']
#         input_dims = config_init['input_dims']
#         device = config_init['device']
#         pmodel = BioV(input_dims, cids, device)
#         pmodel.to(device)
#         return PytorchWrapper(pmodel)
#     return get_biov_helper

# def get_unet(config_init):
#     def get_unet_helper(config, device='cpu'):
#         seed_everything(config_init['seed'])
#         cids = config['cids']
#         model = ModelWrapper(UNet(cids))
#         model.to(device)
#         return model
#     return get_unet_helper

# def get_attention_cnn(config_init):
#     def get_attention_cnn_helper(config, device='cpu'):
#         cnn = get_cnn(config_init)(config, device)
#         cnn_out_dims = [cnn.model.core_subunits, *cnn.model.core[-1].output_dims[1:3], 1]
#         new_readout = LinearAttentionReadout(cnn_out_dims, config['cids'])
#         new_readout.to(device)
#         cnn.model.readout = new_readout         
#         return cnn
#     return get_attention_cnn_helper
