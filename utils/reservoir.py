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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool3d(2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        # if x.shape != skip.shape:
        #     skip = skip[..., :x.shape[-3], :x.shape[-2], :x.shape[-1]]
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        """ Bottleneck """
        self.b = conv_block(256, 512)
        """ Decoder """
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.outputs = nn.Conv3d(64, 1, kernel_size=1, padding=0)
    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.reshape(-1, 1, 35, 35, 24)
            inputs = F.interpolate(inputs, size=(32, 32, 24), mode='trilinear', align_corners=True)
            inputs = inputs[..., -16:]
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
        return outputs