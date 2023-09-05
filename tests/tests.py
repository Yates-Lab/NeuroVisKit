# '''
# This script tests the new regularization modules in foundation.utils.regularization

# It compares them against the NDNT implementation
# '''
# #%%
# #!%load_ext autoreload
# #!%autoreload 2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import utils.regularization as reg
# #%% make fake data

# # make receptive fields
# def make_gabors(dim, NC, sigma=0.5):
#     filters = []
#     for _ in range(NC):
#         # Generate random orientation and frequency
#         orientation = np.random.uniform(0, np.pi)
#         frequency = np.random.uniform(0.5, 1.4)
#         ctrx = np.random.uniform(-1, 1)
#         ctry = np.random.uniform(-1, 1)

#         # Generate Gabor kernel
#         x = np.linspace(-1, 1, dim[0])
#         y = np.linspace(-1, 1, dim[1])
#         t = np.linspace(-1, 1, dim[2])
#         X, Y, T = np.meshgrid(x, y, t)

#         # Rotate coordinates
#         X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
        

#         # Calculate Gabor response
#         gaussian = np.exp(-( (X-ctrx)**2 + (Y-ctry)**2 + T**2) / (2 * (sigma**2)))
#         sinusoid = np.cos(2 * np.pi * frequency * X_rot)

#         kernel = gaussian * sinusoid
#         filters.append(torch.tensor(kernel, dtype=torch.float32))
    
#     return torch.stack(filters)


# #%% make fake data

# NC = 25
# dim = (36,36,24)

# gabor_filters = make_gabors(dim, NC)
# gabor_filters += torch.randn_like(gabor_filters)*0.1

# print("gabor filters shape: ", gabor_filters.shape)
# w = gabor_filters
# keepdims = [0]
# w = w.permute(*[i for i in range(len(w.shape)) if i not in keepdims], *keepdims)
# print("w shape: ", w.shape)

# #%%
# # plot filters
# sx = int(np.ceil(np.sqrt(NC)))
# sy = int(np.ceil(NC/sx))
# plt.figure(figsize=(sx, sy))
# for i in range(NC):
#     plt.subplot(sx,sy,i+1)
#     plt.imshow(gabor_filters[i,:,:,dim[-1]//2].numpy())
#     plt.axis('off')


# ws = gabor_filters.clone().unsqueeze(1)
# print("weights shape: ", ws.shape)

# ws0 = ws.clone()
# ws.requires_grad = True


# from NDNT.modules.regularization import Regularization
# from NDNT.modules.layers import NDNLayer
# # layer = NDNLayer(input_dims = [1] + list(dim), filter_dims=[1] + list(dim), num_filters=NC)



# def step_reg(ws, reg_loss, alpha=0.1):
#     pen = reg_loss(ws)
#     grad = torch.autograd.grad(pen, ws)[0]
#     ws = ws - alpha*grad 
#     return ws, grad, pen

# def run_reg(ws, reg_loss, alpha=0.1, nsteps=100, verbose=True):
#     ws.requires_grad = True
#     for i in range(nsteps):
#         ws, grad, pen = step_reg(ws, reg_loss, alpha)
#         if verbose:
#             print("step: ", i, "penalty: ", pen.item())
#     return ws, grad

# def plot_weights_grad(ws0, ws, grad, cc=0):
#     plt.figure(figsize=(12,4))
#     i = ws0.shape[-1]//2
#     plt.subplot(1,3,1)
#     plt.imshow(ws0[cc, 0, :, :, i].detach().cpu())
#     plt.colorbar()
#     plt.subplot(1,3,2)
#     plt.imshow(ws[cc, 0, :, :, i].detach().cpu())
#     plt.colorbar()
#     plt.subplot(1,3,3)
#     plt.imshow(grad[cc, 0, :, :, i].detach().cpu())
#     plt.colorbar()



# # %% test NDNT regularization: glocalx
# regNDNT = Regularization(filter_dims=list(ws.shape[1:]), vals={'glocalx':10})
# regNDNT.build_reg_modules()

# # inline lambda function that calls the regularization module
# reg_loss = lambda x: regNDNT.compute_reg_loss(torch.flatten(x, start_dim=1).T)

# ws, grad = run_reg(ws0, reg_loss, alpha=0.001, nsteps=500)

# plot_weights_grad(ws0, ws, grad, cc=0)


# # %% test foundation regularization (glocalx)
# import utils.regularization as reg

# ws = ws0.clone().detach()
# ws.requires_grad = True
# regpen = reg.glocalNDNT(coefficient=10, target=ws, dims=[2,3], keepdims=0)

# reg_loss = lambda x: regpen(x)

# ws, grad = run_reg(ws0, reg_loss, alpha=0.001, nsteps=500)

# plot_weights_grad(ws0, ws, grad, cc=0)

# # %% test smoothness

# ws = ws0.clone().detach()
# ws.requires_grad = True
# regpen = reg.laplacian(coefficient=10000, target=ws, dims=[2,3])

# reg_loss = lambda x: regpen(x)

# ws, grad = run_reg(ws0, reg_loss, alpha=0.1, nsteps=500)

# plot_weights_grad(ws0, ws, grad, cc=0)

# #%% Combine smoothness and glocalx
# ws = ws0.clone().detach()
# ws.requires_grad = True
# regpen = reg.Compose(reg.laplacian(coefficient=10000, target=ws, dims=[2,3]),
#                         reg.glocal(coefficient=1, target=ws, dims=[2,3]))

# reg_loss = lambda x: regpen(x)

# ws, grad = run_reg(ws0, reg_loss, alpha=0.001, nsteps=500)

# plot_weights_grad(ws0, ws, grad, cc=0)

# # #%% local Conv

# # ## THIS IS TOO SLOW :(
# # ws = ws0.clone().detach()
# # ws.requires_grad = True
# # regpen = reg.localConv(coefficient=1, target=ws, dims=[2,3], padding='same', padding_mode='constant', normalize=False)

# # reg_loss = lambda x: regpen(x)

# # ws, grad = run_reg(ws0, reg_loss, alpha=0.001, nsteps=10)

# # plot_weights_grad(ws0, ws, grad, cc=0)

# #%% test that regularization doesn't scale with number of filters or upsampling
# NC = 10
# dim = (15,15,24)
# gabor_filters = make_gabors(dim, NC)
# gabor_filters += torch.randn_like(gabor_filters)*0.1

# print("weights shape: ", ws.shape)
# ws0 = gabor_filters.clone().detach().requires_grad_(True)

# ws1 = torch.cat([ws0, ws0], dim=0).detach()
# ws1.requires_grad = True

# regtypes = ['l1', 'l2', 'laplacian', 'local']
# for regtype in regtypes:
#     regpen0 = reg.__dict__[regtype](coefficient=1, target=ws0, dims=[2])
#     regpen1 = reg.__dict__[regtype](coefficient=1, target=ws1, dims=[2])

#     pen0 = regpen0(ws0)
#     pen1 = regpen1(ws1)

#     print("regtype: ", regtype, "pen0: ", pen0.item(), "pen1: ", pen1.item())


# # %%
# from torch.nn.functional import upsample

# # ws0 is gabor filters
# ws0 = gabor_filters.clone().detach().requires_grad_(True)

# # ws1 is upsampled ws0
# ws1 = upsample(ws0, scale_factor=2, mode='bilinear', align_corners=False)

# for regtype in regtypes:
#     regpen0 = reg.__dict__[regtype](coefficient=1, target=ws0, dims=[2])
#     regpen1 = reg.__dict__[regtype](coefficient=1, target=ws1, dims=[2])

#     pen0 = regpen0(ws0)
#     pen1 = regpen1(ws1)

#     print("regtype: ", regtype, "pen0: ", pen0.item(), "pen1: ", pen1.item())

# # %%

# import utils.regularization as reg

# ws = ws0.clone().detach()
# ws.requires_grad = True

# regpen = reg.glocal(coefficient=1, target=ws, dims=[2,3])

# reg_loss = lambda x: regpen(x)

# ws, grad = run_reg(ws0, reg_loss, alpha=0.001, nsteps=500)

# plot_weights_grad(ws0, ws, grad, cc=0)
# # %%