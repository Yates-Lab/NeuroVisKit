from NeuroVisKit._utils.utils import seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models import ModelWrapper
# from models import CNNdense as CNNdenseNDNT
from NeuroVisKit.utils import regularization
from NeuroVisKit.utils import utils
from NeuroVisKit.utils.loss import Poisson
# from unet import UNet, BioV
import math
from tqdm import tqdm
import moten

# from models.cnns import DenseReadout as DenseReadoutNDNT

class ModelWrapper(nn.Module):
    '''
    Instead of inheriting the Encoder class, wrap models with a class that can be used for training
    '''

    def __init__(self,
            model, # the model to be trained
            loss=Poisson(), # the loss function to use
            cids = None, # which units to use during fitting
            **kwargs,
            ):
        
        super().__init__()
        
        if cids is None:
            self.cids = model.cids
        else:
            self.cids = cids
        
        self.model = model
        if hasattr(model, 'name'):
            self.name = model.name
        else:
            self.name = 'unnamed'

        self.loss = loss

    
    def compute_reg_loss(self):
        
        return self.model.compute_reg_loss()

    def prepare_regularization(self, normalize_reg = False):
        
        self.model.prepare_regularization(normalize_reg=normalize_reg)
    
    def forward(self, batch):

        return self.model(batch)

    def training_step(self, batch, batch_idx=None, alternative_loss_fn=None):  # batch_indx not used, right?
        
        y = batch['robs'][:,self.cids]
        y_hat = self(batch)

        if alternative_loss_fn is None:
            if 'dfs' in batch.keys():
                dfs = batch['dfs'][:,self.cids]
                loss = self.loss(y_hat, y, dfs)
            else:
                loss = self.loss(y_hat, y)
        else:
            loss = alternative_loss_fn(y_hat, batch)

        regularizers = self.compute_reg_loss()

        return {'loss': loss.sum() + regularizers, 'train_loss': loss.mean(), 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]
        
        y_hat = self(batch)

        if 'dfs' in batch.keys():
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}

class PytorchWrapper(ModelWrapper):
    def __init__(self, model, *args, cids=None, bypass_preprocess=False, **kwargs):
        super().__init__(model, *args, cids=cids, **kwargs)
        self.bypass_preprocess = bypass_preprocess
        if not hasattr(self, 'cids'):
            self.cids = cids
        self.reg = regularization.extract_reg(self.model, proximal=False)
        self.proximal_reg = regularization.extract_reg(self.model, proximal=True)
        self._lr = None
        print("initialized", self.reg, self.proximal_reg)
    def compute_reg_loss(self, *args, **kwargs):
        loss = sum([r() for r in self.reg]+[0])
        # if type(loss) is int:
        #     return torch.tensor(loss, device=self.parameters().__next__().device).float()
        return loss
    def compute_proximal_reg_loss(self, *args, **kwargs):
        return sum([r() for r in self.proximal_reg]+[0])
    def forward(self, x, pass_dict=False, *args, **kwargs):
        if not self.bypass_preprocess:
            if type(x) is dict and not pass_dict:
                x = x['stim']
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim == 4:
                x = x.unsqueeze(1)
        return self.model(x, *args, **kwargs)
    @property
    def lr(self):
        return self._lr
    @lr.setter
    def lr(self, lr):
        self._lr = lr
        for i in self.proximal_reg:
            if hasattr(i, 'lr'):
                i.lr = lr

class RobsAugmenter(PytorchWrapper):

    def __init__(self, model, *args, gaussian_kernel_size=5, gaussian_kernel_sigma=1, l1=0, l2=0, **kwargs):

        super().__init__(model, *args, **kwargs)

        self.kernel_size = gaussian_kernel_size
        self.kernel_sigma = gaussian_kernel_sigma
        self.l1 = l1
        self.l2 = l2

        self.register_buffer('kernel', utils.gaussian_kernel_1D(gaussian_kernel_size, gaussian_kernel_sigma).unsqueeze(0).unsqueeze(0))

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        
        y = batch['robs'][:,self.cids]
        if self.training:
            ysmooth = F.conv1d(y.T.unsqueeze(1), self.kernel, padding=self.kernel_size//2).squeeze().T
            # y = torch.poisson(ysmooth).detach()
            y = ysmooth.detach() # < torch.rand_like(ysmooth)
        
        y_hat = self(batch)

        if 'dfs' in batch.keys():
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        regularizers = self.compute_reg_loss()
        if self.l1:
            regularizers = regularizers + self.l1*torch.abs(y_hat).mean()
        if self.l2:
            regularizers = regularizers + self.l2*(y_hat**2).mean()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}

def pad_causal(x, layer, kdims=None):
    kdims = kdims if kdims is not None else np.arange(len(layer.kernel_size))
    pad_array = []
    for i in np.arange(len(layer.kernel_size))[::-1]:
        pad_array += [layer.kernel_size[i]-1, 0] if i in kdims else [0, 0]
    return F.pad(x, pad_array)

class DropoutBlock(nn.Module):
    def __init__(self, block, p=0.5):
        super().__init__()
        self.block = block
        self.p = p
        self.original_weights = None
        self.dropout_on = False
        for module in self.block.modules():
            if issubclass(type(module), nn.modules.conv._ConvNd):
                self.c = module.weight
                break
    def forward(self, x):
        if self.training and not self.dropout_on:
            self.dropout()
        x = self.block(x)
        if self.training and self.dropout_on:
            self.undropout()
        return x
    def dropout(self):
        with torch.no_grad():
            self.original_weights = self.c.data
            self.c.data = F.dropout(self.c.data, p=self.p)
            self.dropout_on = True
    def undropout(self):
        with torch.no_grad():
            self.c.data = self.original_weights.to(self.c.data.device)
            self.original_weights = None
            self.dropout_on = False
        
class WindowBlock(nn.Module):
    def __init__(self, block, window_f=lambda x: torch.hamming_window(x, periodic=False)):
        super().__init__()
        self.block = block
        self.original_weights = None
        self.window_on = False
        self.eval_mode = False
        for module in self.block.modules():
            if issubclass(type(module), nn.modules.conv._ConvNd):
                self.c = module.weight
                break
        windows = [window_f(self.c.data.shape[i]) for i in np.arange(len(self.c.data.shape))[2:]]
        self.window_vars = []
        for i, window in enumerate(windows):
            self.register_buffer(f"window{i}", window.to(self.c.data.device)**(1/len(windows)))
            self.window_vars.append(f"window{i}")
        if len(windows) == 1:
            self.einsum_str = "i,noi->noi"
        elif len(windows) == 2:
            self.einsum_str = "i,j,noij->noij"
        elif len(windows) == 3:
            self.einsum_str = "i,j,k,noijk->noijk"
        else:
            raise NotImplementedError("Only up to 3d convs supported")
    def train(self, mode=True):
        if not mode and not self.window_on:
            self.window()
        return super().train(mode)
    def forward(self, x):
        if not self.eval_mode and not self.window_on:
            self.window()
        x = self.block(x)
        if not self.eval_mode and self.window_on:
            self.unwindow()
        return x
    def window(self):
        with torch.no_grad():
            self.original_weights = self.c.data
            self.c.data = torch.einsum(self.einsum_str, *[getattr(self, i) for i in self.window_vars],self.c.data)
            self.window_on = True
    def unwindow(self):
        with torch.no_grad():
            self.c.data = self.original_weights.to(self.c.data.device)
            self.original_weights = None
            self.window_on = False
            
class CBlock3D(nn.Module):
    def __init__(self, cin, cout, k=3, padding="same", nl=F.softplus, window=True):
        super().__init__()
        self.c = nn.Conv3d(cin, cout, kernel_size=k, padding=padding)
        self.is_window = window
        if window:
            self.eval_mode = False
            self.window_on = False
            self.original_weights = None
        self.bn = nn.BatchNorm3d(cout)
        self.nl = nl
    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        x = self.nl(x)
        return x
            
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
        x = self.c(x)
        x = self.bn(x)
        x = self.nl(x)
        return x 
    
class CoreC(nn.Module):
    def __init__(self, input_dims, nl=F.softplus, num_lags=36, window=True):
        super().__init__()
        self.input_dims = input_dims
        self.nl = nl
        self.num_lags = num_lags
        self.c1 = CBlock3D(input_dims[0], 20, k=(num_lags, 11, 11), padding=0)
        self.c2 = nn.Sequential(
            CBlock2D(20, 20, k=11, padding="same"),
            CBlock2D(20, 20, k=11, padding="same"),
            CBlock2D(20, 20, k=11, padding="same"),
        )
        self.regs = regularization.Compose(
            regularization.fourierLocal(1e-3, target=self.c1.c.weight, dims=(2, 3, 4), keepdims=(0, 1)),
        )
    def forward(self, x):
        x = x.squeeze(1).permute(1, 0, 2, 3).unsqueeze(0)
        x = F.pad(x, (5, 5, 5, 5, self.num_lags-1, 0))
        x = self.c1(x)
        x = x.squeeze(0).permute(1, 0, 2, 3)
        return self.c2(x) # shape is (t, 20, h, w)
    
class DenseReadoutC(nn.Module):
    def __init__(self,
        input_dims, 
        num_filters):
        super().__init__()
        self.input_dims=input_dims
        self.num_filters=num_filters
        self.feature = nn.Parameter(utils.KaimingTensor([input_dims[0], num_filters])) # (c, n)
        self.space = nn.Parameter(utils.KaimingTensor([*input_dims[1:], num_filters])) # (x, y, n)
        self.bias = nn.Parameter(torch.zeros([num_filters]))
        self.reg = regularization.Compose(
            regularization.local(1e-1, target=self.space, dims=(0, 1), keepdims=2),
            regularization.l2(1e-7, keepdims=-1, target=self.space),
            regularization.proximalL1(1e-5, keepdims=-1, target=self.feature),
            # regularization.l1(1e-5, keepdims=-1, target=self.feature)
        )
    def forward(self, x):
        return torch.einsum('tcxy,cn,xyn->tn',x,self.feature,self.space) + self.bias
    
class CNNC(nn.Module):
    def __init__(self, input_dims, ncids, nl=F.softplus, output_nl=F.softplus):
        super().__init__()
        print(input_dims[-1], "lags")
        nl = F.relu
        self.core = CoreC(input_dims[:3], nl, input_dims[-1], window=False)
        self.readout = DenseReadoutC([20, *input_dims[1:3]], ncids)
        self.output_nl = output_nl
    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        return self.output_nl(x)
    def fromConfig(config):
        return PytorchWrapper(
            CNNC(config['input_dims'], len(config['cids'])),
            cids=config['cids']
        )

class CNN_time_embed(nn.Module):
    def __init__(self, input_dims, ncids, nl=F.softplus, output_nl=F.softplus):
        super().__init__()
        print(input_dims[-1], "lags")
        self.core = nn.Sequential(
            CBlock2D(input_dims[-1], 20, k=11, padding="same", nl=nl),
            CBlock2D(20, 20, k=11, padding="same", nl=nl),
            CBlock2D(20, 20, k=11, padding="same", nl=nl),
            CBlock2D(20, 20, k=11, padding="same", nl=nl),
        )
        self.readout = DenseReadoutC([20, *input_dims[1:3]], ncids)
        self.output_nl = output_nl
    def forward(self, x):
        x = x.squeeze(1).permute(0, 3, 1, 2)
        x = self.core(x)
        x = self.readout(x)
        return self.output_nl((x-0.4)/0.4)
    def fromConfig(config):
        return PytorchWrapper(
            CNN_time_embed(config['input_dims'], len(config['cids'])),
            cids=config['cids']
        )

# class CNNdense(nn.Module):
#     def __init__(self, input_dims, cids, nl=F.softplus, output_nl=F.softplus):
#         super().__init__()
#         num_filters = [20, 20, 20, 20]
#         filter_width = [11, 11, 11, 11]
#         scaffold = [len(num_filters)-1]
#         num_inh = [0]*len(num_filters)
#         modifiers = None
#         self.model = CNNdenseNDNT(
#             input_dims,
#             num_filters,
#             filter_width,
#             scaffold,
#             num_inh,
#             is_temporal=False,
#             NLtype='softplus',
#             batch_norm=True,
#             norm_type=0,
#             noise_sigma=0,
#             NC=len(cids),
#             bias=False,
#             reg_core=None,
#             reg_hidden=None,
#             reg_readout={'glocalx':.1, 'l2':0.1},
#             reg_vals_feat={'l1':0.01},
#             cids=cids,
#             modifiers=modifiers,
#             window='hamming')
#     def fromConfig(config):
#         model = CNNdense(
#             config['input_dims'],
#             config['cids']
#         ).model
#         model.prepare_regularization()
#         return ModelWrapper(model)

# MODEL_DICT = {
#     'CNNdense': get_cnn,
#     # 'UNet': get_unet,
#     # 'AttentionCNN': get_attention_cnn,
#     # 'BioV': get_biov,
#     # 'gaborC': get_gaborC,
#     # 'seperableCNN': get_separable_cnn,
#     'cnnc': get_cnnc,
#     'cnn_time_embed': get_cnn_time_embed,
# }
    
    
# def get_model(config):
#     if not factory:
#         return MODEL_DICT[config['model']](config)(config)
#     return MODEL_DICT[config['model']](config)

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


# def gain(x, dim):
#     alpha = 1
#     x_sub = 1 + (torch.abs(x)**alpha).sum(dim=dim, keepdim=True)
#     return x / x_sub

# def dl_to_data(dl):
#     return 

# class gaborC(nn.Module):
#     #assumes input has been preprocessed with gabors
#     def __init__(self, cids, nl=nn.Softplus()):
#         super().__init__()
#         self.nfilters = 28084#moten.get_default_pyramid(vhsize=(70, 70), fps=45).nfilters
#         self.cids = cids
#         self.linear = nn.Linear(self.nfilters, len(cids))
#         # self.linear.weight.data = torch.zeros_like(self.linear.weight.data)
#         # self.linear.bias.data = torch.zeros_like(self.linear.bias.data)
#         self.lag = 0
#         self.output_NL = nl
#     def forward(self, x):
#         return self.output_NL(self.linear(x["stim"]))
#     def compute_reg_loss(self, *args, **kwargs):
#         return torch.abs(self.linear.weight).mean()*10
#     def prepare_model(self, train_dl, *args, **kwargs):
#         if True:
#             data = {k: torch.cat([d[k] for d in train_dl]) for k in next(iter(train_dl)).keys()}
#             nfilters = data["stim"].shape[1]
#             stim, robs, dfs = data["stim"], data["robs"], data["dfs"]
#             stas = torch.zeros((65, nfilters), device=stim.device)
#             n = torch.zeros((65), device=stim.device)
#             # stim = (stim-stim.mean(0))/stim.std(0)
#             for i in range(len(robs)):
#                 weights = (robs[i]*dfs[i]).reshape(-1, 1)
#                 corr = stim[i].unsqueeze(0)
#                 stas += corr*weights
#                 n+=dfs[i]
#             self.linear.weight.data[:, :] = (stas / n.reshape(65, 1))[self.cids, :].to(stim.device)/100
            
#     def get_gaborC(config_init):
#         def get_gaborC_helper(config, device='cpu'):
#             seed_everything(config_init['seed'])
#             model = PytorchWrapper(gaborC(config['cids']), bypass_preprocess=True)
#             if 'lightning' not in config_init or not config_init['lightning']:
#                 model.to(device)
#             return model
#         return get_gaborC_helper