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

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=k, padding="same")
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=k, padding="same")
        self.bn2 = nn.BatchNorm3d(out_c)
        self.selu = nn.SELU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.selu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu(x)
        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.conv = conv_block(in_c, out_c, k=k)
        self.pool = nn.MaxPool3d(2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, k=k)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        # if x.shape != skip.shape:
        #     skip = skip[..., :x.shape[-3], :x.shape[-2], :x.shape[-1]]
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self, outc=62):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(1, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        """ Bottleneck """
        self.b = conv_block(128, 256)
        """ Decoder """
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)
        """ Classifier """
        self.outputs = nn.Conv3d(32, outc, kernel_size=1, padding=0)
    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.reshape(-1, 1, 35, 35, 24)
        inputs = F.interpolate(inputs, size=(32, 32, 32), mode='trilinear', align_corners=True)
            # inputs = inputs[..., -16:]
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        """ Bottleneck """
        b = self.b(p3)
        """ Decoder """
        d2 = self.d2(b, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return F.interpolate(outputs, size=(35, 35, 24), mode='trilinear', align_corners=True)
    
def residual(y, x):
    if x.shape[1] > y.shape[1]:
        x = x[:, :y.shape[1]]
    elif x.shape[1] < y.shape[1]:
        return torch.cat((y[:, :x.shape[1]] + x, y[:, x.shape[1]:]), dim=1)
    return y + x

class simple_conv_block(nn.Module):
    def __init__(self, in_c, out_c, k=3, groups=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=k, padding="same", groups=groups)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=k, padding="same", groups=groups)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.selu = nn.SELU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.selu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu(x)
        return residual(x, inputs)
    
class simple_encoder_block(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.conv = simple_conv_block(in_c, out_c, k=k)
        self.pool = nn.MaxPool3d(2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class simple_decoder_block(nn.Module):
    def __init__(self, in_c, out_c, skip_c, k=3):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = simple_conv_block(out_c+skip_c, out_c, k=k)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Bias(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))
    def forward(self, inputs):
        return inputs + self.bias
class LinearBasis(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.basis = nn.Linear(inp, out)
    def forward(self, inputs):
        return self.basis(inputs)

def timeIsHidden(fn, x):
    '''
        Hide time dimension.
    '''
    b, c, t, h, w = x.shape
    x = rearrange(x, "b c t h w -> (b t) c h w", b=b, t=t, c=c, h=h, w=w)
    x = fn(x)
    x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
    return x

def foldTime(x):
    b, c, t, h, w = x.shape
    return rearrange(x, "b c t h w -> b (c t) h w", b=b, t=t, c=c, h=h, w=w)

class HierarchicalAdapterBlock(nn.Module):
    def __init__(self, outc):
        # dim_out must be a power of 2
        assert math.log2(outc) == int(math.log2(outc))
        self.preprocessor = nn.Sequential(
            nn.Conv3d(1, 3, 3, padding="same"),
            nn.SELU(),
        )
        levels = [2**i for i in range(int(math.log2(outc)-1))] # 1, 2, 4, 8
        kernel_sizes = [1] + [i+1 for i in levels[1:]] # 1, 3, 5, 9
        self.hierarchical = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(3, 3),
                nn.Conv3d(3, levels[-i], kernel=(1, kernel_sizes[i], kernel_sizes[i]), groups=3, padding="same"),
                nn.SELU(),
            ) for i in range(len(levels))
        ])
        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(1, outc),
                nn.Conv3d(outc, levels[-i], kernel=(kernel_sizes[i], 1, 1), padding="same"),
                nn.SELU(),
            ) for i in range(len(levels))
        ])
    def forward(self, x):
        x = self.preprocessor(x)
        x = torch.cat([x]+[model(x) for model in self.hierarchical], dim=1)
        x = torch.cat([x]+[model(x) for model in self.temporal], dim=1)
        return x
        
    
class build_hnet(nn.Module):
    def __init__(self, outc=64):
        super().__init__()
        self.hnet = HierarchicalAdapterBlock(64)
        
class build_simple_resnet(nn.Module):
    def __init__(self, outc=62):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.e1 = simple_conv_block(1, 16, k=9)
        self.e2 = simple_conv_block(16, 32, k=5)
        self.e3 = simple_conv_block(32, 64, k=5)
        self.b = nn.Sequential(
            conv_block(64, 128, k=3),
            nn.Flatten(),
            nn.Linear(128*4*4*4, 1240),
        )
        self.basis = LinearBasis(20, 32*32*32)
        self.readout = nn.Sequential(
            Rearrange('b (n z) -> (b n) z', z=20, n=62),
            self.basis,
            Rearrange('(b n) z-> b (n z)', z=32*32*32, n=62),
            Bias(62*32*32*32),
            nn.Tanh()
        )
        # self.readout = nn.Sequential(
        #     nn.ConvTranspose3d(62, 64, kernel_size=2, stride=2, padding=0),
        #     simple_conv_block(64, 64, k=5, groups=16),
        #     nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, padding=0, groups=16),
        #     simple_conv_block(64, 64, k=5, groups=32),
        #     nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, padding=0, groups=32),
        #     simple_conv_block(64, 64, k=9, groups=32),
        #     nn.Conv3d(64, outc, kernel_size=1, padding=0),
        #     nn.Tanh()
        # )
    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.reshape(-1, 1, 35, 35, 24)
        inputs = F.interpolate(inputs, size=(32, 32, 32), mode='trilinear', align_corners=True)
        s1 = self.pool(self.e1(inputs))
        s2 = self.pool(self.e2(s1))
        s3 = self.pool(self.e3(s2))
        b = self.b(s3)
        outputs = self.readout(b).reshape(-1, 62, 32, 32, 32)
        return F.interpolate(outputs, size=(35, 35, 24), mode='trilinear', align_corners=True)

class HyperGLM(nn.Module):
    def __init__(self, cids):
        super().__init__()
        self.unet = build_simple_resnet(len(cids))
        self.output_NL = F.softplus
        self.cids = cids
        self.scale = nn.Parameter(torch.ones(1, len(cids)))
        self.bias = nn.Parameter(torch.zeros(1, len(cids)))
    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.reshape(-1, 1, 35, 35, 24)
        rates = torch.einsum("bxyt,bcxyt->bc", inputs.squeeze(1), self.unet(inputs))
        return self.output_NL(rates * self.scale + self.bias)
    def irf(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.reshape(-1, 1, 35, 35, 24)
        return self.unet(inputs)
        