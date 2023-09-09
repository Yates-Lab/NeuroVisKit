'''
This script tests the new regularization modules in foundation.utils.regularization

It compares them against the NDNT implementation
'''
#%%
#!%load_ext autoreload
#!%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1),  # green with alpha = 50%
    "savefig.facecolor": (1.0, 1.0, 1.0, 1),  # blue  with alpha = 20%
})

#%% make fake data

# make receptive fields
def make_gabors(dim, NC, sigma=0.5):
    filters = []
    for _ in range(NC):
        # Generate random orientation and frequency
        orientation = np.random.uniform(0, np.pi)
        frequency = np.random.uniform(0.5, 1.4)
        ctrx = np.random.uniform(-1, 1)
        ctry = np.random.uniform(-1, 1)

        # Generate Gabor kernel
        x = np.linspace(-1, 1, dim[0])
        y = np.linspace(-1, 1, dim[1])
        t = np.linspace(-1, 1, dim[2])
        X, Y, T = np.meshgrid(x, y, t)

        # Rotate coordinates
        X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
        

        # Calculate Gabor response
        gaussian = np.exp(-( (X-ctrx)**2 + (Y-ctry)**2 + T**2) / (2 * (sigma**2)))
        sinusoid = np.cos(2 * np.pi * frequency * X_rot)

        kernel = gaussian * sinusoid
        filters.append(torch.tensor(kernel, dtype=torch.float32))
    
    return torch.stack(filters)

def step_reg(ws, reg, alpha=0.1, proximal=False):
    if proximal:
        grad = ws.detach().clone()
        pen = reg()
        grad = ws.detach().clone() - grad
    else:
        pen = reg()
        grad = torch.autograd.grad(pen, reg.target)[0]
        ws = ws - alpha*grad 
    return ws, grad, pen

def run_reg(ws, reg, alpha=0.1, nsteps=100, verbose=True, proximal=False):
    ws.requires_grad = True
    for i in range(nsteps):
        ws, grad, pen = step_reg(ws, reg, alpha, proximal=proximal)
        if verbose:
            print("type:", reg.__name__, "step: ", i, "penalty: ", pen.item())
    return ws, grad

def plot_weights_grad(ws0, ws, grad, cc=0):
    plt.figure(figsize=(12,4))
    i = ws0.shape[-1]//2
    plt.subplot(1,3,1)
    plt.imshow(ws0[cc, 0, :, :, i].detach().cpu())
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(ws[cc, 0, :, :, i].detach().cpu())
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(grad[cc, 0, :, :, i].detach().cpu())
    plt.colorbar()



#%% make fake data
NC = 25
dim = (36,36,24)

gabor_filters = make_gabors(dim, NC)
gabor_filters += torch.randn_like(gabor_filters)*0.1

print("gabor filters shape: ", gabor_filters.shape)
w = gabor_filters
keepdims = [0]
w = w.permute(*[i for i in range(len(w.shape)) if i not in keepdims], *keepdims)
print("w shape: ", w.shape)

# plot filters
sx = int(np.ceil(np.sqrt(NC)))
sy = int(np.ceil(NC/sx))
plt.figure(figsize=(sx, sy))
for i in range(NC):
    plt.subplot(sx,sy,i+1)
    plt.imshow(gabor_filters[i,:,:,dim[-1]//2].numpy())
    plt.axis('off')


ws = gabor_filters.clone().unsqueeze(1)
print("weights shape: ", ws.shape)

ws0 = ws.clone()
ws.requires_grad = True

#%%



# %% test foundation regularization (glocalx)
import NeuroVisKit.utils.regularization as reg
import time
regs = reg.get_regs_dict()
print("detected regs: ", regs.keys())
nsteps = 450
for k, v in regs.items():
    ws = ws0.clone().detach()
    regpen = v(target=ws, dims=[2,3], keepdims=0)
    stime = time.time()
    if regpen._parent_class.__name__ == "RegularizationModule":
        ws, grad = run_reg(ws, regpen, alpha=1e-1, nsteps=nsteps, proximal=False)
    elif regpen._parent_class.__name__ == "ProximalRegularizationModule":
        ws, grad = run_reg(ws, regpen, alpha=1, nsteps=nsteps, proximal=True)
    stime = (time.time() - stime)/nsteps
    plot_weights_grad(ws0, ws, grad, cc=0)
    plt.gcf().suptitle(f"{k} ({stime:.2f} s/step)")
    plt.gcf().tight_layout()
    plt.show()
    del ws, regpen, grad
        
#%%
# ws = ws0.clone().detach()
# ws.requires_grad = True
# regpen = reg.local(coefficient=.0001, target=ws, dims=[2,3])

# reg_loss = lambda x: regpen(x)

# ws, grad = run_reg(ws0, reg_loss, alpha=0.1, nsteps=5500)

# plot_weights_grad(ws0, ws, grad, cc=0)
