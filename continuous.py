
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from models import CNNdense
# from tqdm import tqdm
# import torch.nn.functional as F
# from models.shifters import shift_im
# from NDNT.modules.layers import NDNLayer, STconvLayer, ConvLayer

# def get_model(model_name, input_dims, cids, is_temporal=2):

#     if model_name=='CNN':

#         # parameters of architecture
#         num_filters = [20, 20, 20, 20]
#         filter_width = [9, 9, 9, 9]
#         num_inh = [0]*len(num_filters)
#         scaffold = [len(num_filters)-1]

#         cr0 = CNNdense(input_dims,
#             num_subunits=num_filters,
#             filter_width=filter_width,
#             num_inh=num_inh,
#             cids=cids,
#             bias=False,
#             scaffold=scaffold,
#             padding='valid',
#             is_temporal=is_temporal,
#             batch_norm=True,
#             window='hamming',
#             norm_type=0,
#             reg_core=None,
#             reg_hidden=None,
#             reg_readout={'glocalx':1, 'l1':0.1},
#             reg_vals_feat={'l1':0.1},
#                         )
#         cr0.name = 'CNNdense'

#     # elif model_name=='LGNReccurent':



#     return cr0

# def radial_mean(I, nbins=50):
#     sx, sy = I.shape
#     X, Y = np.ogrid[0:sx, 0:sy]
    
#     r = np.hypot(X - sx/2, Y - sy/2)

#     rbin = (nbins* r/r.max()).astype(np.int)

    
#     rmean = np.array([I[rbin==ri].mean().item() for ri in range(nbins)])
#     return rmean

# def calc_power_spectrum(ds, inds, start):
#     powerspectrum = 0
#     for i in tqdm(inds):
#         batch = ds[i]
#         # get spatiotemporal movie
#         I = batch['stim'][start:,...].squeeze() - batch['stim'][start:,...].mean()
        
#         # size of this movie
#         sz = I.shape # [T, H, W]
#         # window function
#         win = torch.from_numpy(np.einsum('x,y,t->xyt', np.hamming(sz[0]), np.hamming(sz[1]), np.hamming(sz[2])))
        
#         # compute power spectrum
#         F = torch.fft.fftshift(torch.fft.fftn(I*win, s=(256,128,128)))
#         P = torch.abs(F)**2
#         # running sum
#         powerspectrum += P

#     powerspectrum /= len(inds)
#     return powerspectrum

# def check_spikes_per_batch(ds):
#     nspikes = []
#     print("counting spikes in each fixation")
#     for ifix in tqdm(range(len(ds))):
#         batch = ds[ifix]
#         nspikes.append(batch['robs'].mean().item())
        
#     plt.plot(np.stack(nspikes), '.')
#     plt.ylim(0, 1)
#     if ds.use_blocks:
#         plt.xlabel('block number')
#     else:
#         plt.xlabel('fixation number')

#     plt.ylabel('proportion of bins with spikes')

#     sus_fix = np.where(np.stack(nspikes)>.2)[0]
#     if len(sus_fix)==0:
#         print('no fixations with bad spikes')
#         sus_fix = np.arange(len(ds))

#     return nspikes, sus_fix

# def check_raw_sta(ds, lag=7, stim_id=1):
#     Stim = ds.fhandles[0][ds.requested_stims[stim_id]][ds.stimset]['Stim'][:,:,:].astype(np.float32)
#     Stim = torch.from_numpy(Stim)
#     frame_times = ds.fhandles[0][ds.requested_stims[stim_id]][ds.stimset]['frameTimesOe'][0,:]
#     labels = ds.fhandles[0][ds.requested_stims[stim_id]][ds.stimset]['labels'][0,:]

#     # get spike times
#     st = ds.fhandles[0]['Neurons'][ds.spike_sorting]['times'][0,:]
#     clu = ds.fhandles[0]['Neurons'][ds.spike_sorting]['cluster'][0,:].astype(int)

#     # plt.figure(figsize=(10,5))
#     # plt.plot(st, clu, 'k|', ms=.1)
#     # plt.plot(frame_times, labels, 'r')

#     bin_size = 8e-3
#     ind1 = np.digitize(st, frame_times)
#     ind2 = np.digitize(st, frame_times+bin_size)

#     ix = ind2-ind1 != 0
#     # ix = np.logical_and(ix, ind2+1 == ind1)
#     # ix = np.logical_and(ix,clu>0)

#     robs = torch.sparse_coo_tensor( np.asarray([ind1[ix]-1, clu[ix]-1]),
#             np.ones(len(clu[ix])), (len(frame_times), np.max(clu)) , dtype=torch.float32)
#     robs = robs.to_dense()
#     robs = robs[:,robs.sum(dim=0)>0]
#     NC = robs.shape[1]
#     # inds = np.where(np.logical_and(labels==1, np.arange(robs.shape[0])>lag))[0]
#     inds = np.where(np.arange(robs.shape[0])>lag)[0]
#     sta = torch.einsum('hwt, tn->hwn', Stim[:,:,inds-lag], robs[inds,:]-robs[inds,:].mean(dim=0))

#     plt.figure(figsize=(10,10))
#     sx = int(np.ceil(np.sqrt(NC)))
#     sy = int(np.ceil(NC/sx))
#     for cc in range(NC):
#         plt.subplot(sx,sy,cc+1)
#         plt.imshow(sta[:,:,cc])

#     return sta

# '''' MODEL STUFF'''

# def pseudo_derivative(v_scaled, dampening_factor=0.3):
#     '''
#     translated from Bellec et al., 2021
#     code example at https://github.com/EPFL-LCN/pub-bellec-wang-2021-sample-and-measure/blob/main/src/buildnet.py
#     '''
#     return dampening_factor * torch.maximum(0.,1 - torch.abs(v_scaled))

# class SpikeFunction(torch.autograd.Function):
#     """
#     Here we implement our spiking nonlinearity which also implements 
#     the surrogate gradient. By subclassing torch.autograd.Function, 
#     we will be able to use all of PyTorch's autograd functionality.
#     """
    
#     scale = 100.0 # controls steepness of surrogate gradient

#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we sample spikes from the input Tensor
#         and return it. ctx is a context object that we use to stash information which 
#         we need to later backpropagate our error signals. To achieve this we use the 
#         ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input)
#         out = torch.greater(torch.sigmoid(input), torch.rand_like(input))
        
#         return out.float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor we need to compute the 
#         surrogate gradient of the loss with respect to the input. 
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()

#         dE_dz = grad_input
#         dz_dv_scaled = pseudo_derivative(input)
#         dE_dv_scaled = dE_dz * dz_dv_scaled
        
#         return dE_dv_scaled
    

# class SurrGradSpike(torch.autograd.Function):
#     """
#     Here we implement our spiking nonlinearity which also implements 
#     the surrogate gradient. By subclassing torch.autograd.Function, 
#     we will be able to use all of PyTorch's autograd functionality.
#     Here we use the normalized negative part of a fast sigmoid 
#     as this was done in Zenke & Ganguli (2018).
#     """
    
#     scale = 100.0 # controls steepness of surrogate gradient

#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we compute a step function of the input Tensor
#         and return it. ctx is a context object that we use to stash information which 
#         we need to later backpropagate our error signals. To achieve this we use the 
#         ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input)
#         out = torch.zeros_like(input)
#         out[input > 0] = 1.0
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor we need to compute the 
#         surrogate gradient of the loss with respect to the input. 
#         Here we use the normalized negative part of a fast sigmoid 
#         as this was done in Zenke & Ganguli (2018).
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
#         return grad

    

# # build convolutional block
# class ConvBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
        
#         return x
    
# class TCN(nn.Module):

#     def __init__(self, in_channels=1, ks=[5,3,2,2,2], nc=[4,4,8,16,32]):

#         super(TCN, self).__init__()

#         # convolutional layers
#         self.core = nn.Sequential()
#         self.core.add_module('conv_block_{}'.format(0), ConvBlock(in_channels, nc[0], ks[0], groups=in_channels))

#         self.padding = ks[0]-1    
#         for i in range(1,len(ks)):
#             self.core.add_module('conv_block_{}'.format(i), ConvBlock(nc[i-1], nc[i], ks[i]))
#             self.padding += ks[i] - 1

#         self.num_outputs = nc[-1]

#     def forward(self, x):
#         x = self.core(x)
#         return x
    
# # make CNN
# class CNN(nn.Module):

#     def __init__(self, in_channels=1, ks=[5,3,2,2,2], nc=[4,4,8,16,32]):

#         super(CNN, self).__init__()

#         # convolutional layers
#         self.core = nn.Sequential()
#         self.core.add_module('conv_block_{}'.format(0), ConvBlock(in_channels, nc[0], ks[0], groups=in_channels))

#         self.padding = ks[0]-1    
#         for i in range(1,len(ks)):
#             self.core.add_module('conv_block_{}'.format(i), ConvBlock(nc[i-1], nc[i], ks[i]))
#             self.padding += ks[i] - 1

#         self.num_outputs = nc[-1]

#     def forward(self, x):
#         x = self.core(x)
#         return x
    
# # make factorized readout
# class FactorizedReadout(nn.Module):

#     def __init__(self,
#         input_dims, 
#         num_filters,
#         reg_vals={'glocalx': 10, 'edge_x': 10, 'center': .1},
#         reg_vals_feat={'l1':0.01, 'l2':0.01},
#         window='hamming',
#         **kwargs):

#         super().__init__()
        
#         assert len(input_dims)==4, 'FactorizedReadout: input_dims must have form [channels, height, width, 1]'
#         assert input_dims[-1]==1, 'Factorized cannot have time lags'

#         self.input_dims=input_dims
#         self.num_filters=num_filters
#         in_channels = int(input_dims[0])
#         spat_dims = [1] + input_dims[1:] 
#         self.feature = NDNLayer(input_dims=[in_channels, 1, 1, 1], num_filters=num_filters, reg_vals=reg_vals_feat, window=window, **kwargs)
#         self.space = NDNLayer(input_dims=spat_dims, num_filters=num_filters, reg_vals=reg_vals, **kwargs)

#         self.build_reg_modules()
    
#     def forward(self, x):
        
#         # get weights
#         wf = self.feature.preprocess_weights()
#         ws = self.space.preprocess_weights()
#         ws = ws.view(self.space.filter_dims[1:3] + [self.space.num_filters])
#         # process input dims
#         x = x.view([-1]+self.input_dims)

#         return torch.einsum('bcxyl,xyn,cn->bn',x,ws,wf)

#     def build_reg_modules(self, normalize_reg=False):
        
#         self.space.reg.normalize = normalize_reg
#         self.feature.reg.normalize = normalize_reg
#         self.space.reg.build_reg_modules()
#         self.feature.reg.build_reg_modules()

#     def compute_reg_loss(self):
        
#         rloss = self.feature.compute_reg_loss()
#         rloss += self.space.compute_reg_loss()

#         return rloss
            

# # make RNN
# class sRNN(nn.Module):

#     def __init__(self, input_size, output_size):

#         super(sRNN, self).__init__()

#         self.input_size = input_size
#         self.output_size = output_size

#         self.w1 = nn.Linear(input_size, output_size, bias=False)
#         self.w2 = nn.Linear(output_size, output_size, bias=False)

#         self.spike_fn  = SpikeFunction.apply

#     def forward(self, inputs):
        
#         # map inputs to dynamical state
#         h1 = self.w1(inputs)

#         # initialize the synaptic currents and membrane potentials
#         mem = torch.ones_like(h1[0,:])*-10.0
#         mem_rec = []

#         # spiking is generated from dynamics
#         spk_rec = []
        
#         # initialize spiking
        

#         # Compute membrane dynamics
#         for t in range(inputs.shape[0]):
            
#             # generate spiking
#             mthr = mem-1.0
#             out = self.spike_fn(mthr)

#             mem_rec.append(mem)
#             spk_rec.append(out)
            
#             # dynamics
#             new_mem = self.w2(mem) + h1[t,:]

#             mem = new_mem

#         mem_rec = torch.stack(mem_rec,dim=0)
#         spk_rec = torch.stack(spk_rec,dim=0)

#         return mem_rec, spk_rec
    
# class CNNRNN(nn.Module):

#     def __init__(self, cnn, rnn, readout):

#         super().__init__()
#         self.cnn = cnn
#         self.rnn = rnn
#         self.readout = readout

#     def forward(self, x):
            
#         x = self.cnn(x)
#         x = self.readout(x)
#         x,_ = self.rnn(x)

#         return x

# class Shifter(nn.Module):
#     '''
#     Shifter wraps a model with a shifter network
#     '''

#     def __init__(self, model,
#         num_hidden=20,
#         affine=False,
#         **kwargs):

#         super().__init__()

#         self.model = model
#         self.affine = affine
#         self.name = 'shifter'

#         if affine:
#             self.shifter = nn.Sequential(
#                 nn.Linear(2, num_hidden, bias=False),
#                 nn.Softplus(), 
#                 nn.Linear(num_hidden, 4, bias=True))
#         else:
#             self.shifter = nn.Sequential(
#                 nn.Linear(2, num_hidden, bias=False),
#                 nn.Softplus(),
#                 nn.Linear(num_hidden, 2, bias=True))
        
#         # dummy variable to pass in as the eye position for regularization purposes
#         self.register_buffer('reg_placeholder', torch.zeros(1,2))
    
#     def shift_stim(self, stim, shift):
#         '''
#         flattened stim as input
#         '''
#         return shift_im(stim, shift, self.affine)

#     def compute_reg_loss(self):
        
#         rloss = self.model.compute_reg_loss()
#         rloss += self.shifter(self.reg_placeholder).abs().sum()*10
        
#         return rloss

#     def forward(self, stim, eyepos):
#         '''
#         The model forward calls the existing model forward after shifting the stimulus
#         That's it.
#         '''
#         # calculate shift
#         shift = self.shifter(eyepos)

#         # replace stimulus in batch with shifted stimulus
#         stim = self.shift_stim(stim, shift)
#         # call model forward
#         return self.model(stim)

# class TCN(nn.Module):

#     def __init__(self, input_dims=[1, 70, 70], num_lags=36, ks=[7, 9, 9, 9], nc=[20, 20, 20, 20]):

#         super(TCN, self).__init__()

#         # convolutional layers
#         self.core = nn.Sequential()
#         c1 = STconvLayer(input_dims=input_dims + [num_lags],
#                     num_filters=nc[0],
#                     filter_dims=ks[0],
#                     norm_type=0,
#                     num_inh = 0,
#                     window='hamming',
#                     padding='valid',
#                     initialize_center=False,
#                     NLtype='relu',
#                     output_norm='batch',
#                     reg_vals={'glocalx':.001, 'd2t':.001})
        
#         self.core.add_module('conv_{}'.format(0), c1)

#         for i in range(1,len(ks)):
#             self.core.add_module('conv_{}'.format(i), 
#                         ConvLayer(input_dims=self.core[i-1].output_dims,
#                         num_filters=nc[i],
#                         filter_dims=ks[i],
#                         norm_type=0,
#                         num_inh = 0,
#                         window='hamming',
#                         padding='valid',
#                         initialize_center=True,
#                         NLtype='relu',
#                         output_norm='batch',
#                         reg_vals={'glocalx': .1})
#                                     )
            
#         self.num_outputs = nc[-1]

#     def forward(self, x):
#         x = self.core(x)
#         return x

#     def build_reg_modules(self):
#         for c in self.core:
#             c.reg.build_reg_modules()
    
#     def compute_reg_loss(self):
#         rloss = 0
#         for i in range(len(self.core)):
#             rloss += self.core[i].compute_reg_loss()
#         return rloss
    
# class CoreReadout(nn.Module):

#     def __init__(self, cnn, readout):

#         super().__init__()
#         self.cids = list(range(readout.num_filters))
#         self.core = cnn
#         self.readout = readout
#         self.readout.build_reg_modules()
#         self.core.build_reg_modules()
#         self.bias = nn.Parameter(torch.zeros(readout.num_filters))
#         self.nl = nn.Softplus()
#         self.vthr = 0.4

#     def forward(self, batch):

#         x = self.core(batch['stim'])
#         x = self.readout(x)
#         x = x + self.bias

#         return self.nl( (x - self.vthr) / self.vthr)
    
#     def prepare_regularization(self, normalize_reg=None):
#         self.core.build_reg_modules()
#         self.readout.build_reg_modules()
    
#     def compute_reg_loss(self):
#         rloss = self.core.compute_reg_loss()
#         rloss += self.readout.compute_reg_loss()
        
#         return rloss
    

# class MaskedLoss(nn.Module):

#     def __init__(self, loss_fn, norm=1):

#         super().__init__()
#         self.loss_fn = loss_fn
#         self.unit_loss = loss_fn
#         self.norm = norm

#     def forward(self, yhat, y, dfs=None):
#         loss = self.loss_fn(yhat, y)

#         if dfs is not None:
#             loss = loss * dfs
#             if self.norm==0:
#                 n = dfs.sum(dim=0).clamp(1.0)
#             elif self.norm==1:
#                 n = (y*dfs).sum(dim=0).clamp(1.0)
#         else:
#             if self.norm==0:
#                 n = y.shape[0]
#             elif self.norm==1:
#                 n = y.sum(dim=0).clamp(1.0)

#         loss = loss.sum(dim=0)
#         loss = loss / n
#         # loss = 10*loss / torch.maximum(y.sum(dim=0), torch.tensor(1.0))

#         return loss.mean()
    
# class Corr(nn.Module):
#     def __init__(self, eps=1e-12, detach_target=True):
#         """
#         Compute correlation between the output and the target

#         Args:
#             eps (float, optional): Used to offset the computed variance to provide numerical stability.
#                 Defaults to 1e-12.
#             detach_target (bool, optional): If True, `target` tensor is detached prior to computation. Appropriate when
#                 using this as a loss to train on. Defaults to True.
#         """
#         self.eps = eps
#         self.detach_target = detach_target
#         super().__init__()

#     def forward(self, output, target):
#         if self.detach_target:
#             target = target.detach()
#         delta_out = output - output.mean(0, keepdim=True)
#         delta_target = target - target.mean(0, keepdim=True)

#         var_out = delta_out.pow(2).mean(0, keepdim=True)
#         var_target = delta_target.pow(2).mean(0, keepdim=True)

#         corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
#             (var_out + self.eps) * (var_target + self.eps)
#         ).sqrt()
#         return (1-corrs).mean()
    
