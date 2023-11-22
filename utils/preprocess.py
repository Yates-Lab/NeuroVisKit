#%%
from NeuroVisKit._utils.utils import time_embedding
import torch
from NeuroVisKit._utils.utils import print_off
import torch.nn as nn
import inspect

def get_process_dict():
    return {
        k:v for k,v in globals().items()
        if k[0].isupper()
    }
    
def Embed(x, num_lags=24):
    if "embedded" in x:
        return x
    x["stim"] = time_embedding(x["stim"], num_lags)
    x["dfs"] = x["dfs"][num_lags-1:]
    x["robs"] = x["robs"][num_lags-1:]
    x["embedded"] = True
    return x

def Binarize(x):
    x["robs"][x["robs"] > 0] = 1
    return x

class GaborPreprocess(nn.Module):
    def __init__(self, hw, fps=240):
        super().__init__()
        import moten
        self.hw = hw
        self.pyramid = moten.get_default_pyramid(vhsize=hw, fps=fps)
    def forward(self, x):
        bsize = x["stim"].shape[0]
        with print_off():
            x["stim"] = self.pyramid.project_stimulus(x["stim"].reshape(bsize, *self.hw).cpu().numpy())
        x["stim"] = torch.from_numpy(x["stim"].reshape(bsize, -1)).to(x["robs"].device)
        return x
    
class Trim(nn.Module):
    def forward(self, x):
        if "trimmed" in x:
            return x
        x["robs"] = x["robs"][35:]
        x["dfs"] = x["dfs"][35:]
        x["trimmed"] = True
        return x