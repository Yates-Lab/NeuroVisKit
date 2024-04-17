'''
Collection of tools for analyzing and plotting RFs
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def get_sta(stim, robs, dfs=None, modifier= lambda x: x, inds=None, lags=[8], batchsize=None, device=None):
    '''
    Compute the STA for a given stimulus and response
    stim: [N, C, H, W] tensor
    robs: [N, NC] tensor
    dfs: mask for included samples
    inds: indices to use for the analysis
    modifier: function to apply to the stimulus before computing the STA
    lags: list of lags to compute the STA over

    returns: [NC, C, H, W, len(lags)] tensor
    '''

    if device is None:
        device = stim.device

    if isinstance(lags, int):
        lags = [lags]
    
    if inds is None:
        inds = np.arange(stim.shape[0])

    NT = stim.shape[0]
    sz = list(stim.shape[1:])
    NC = robs.shape[1]
    sta = torch.zeros( [NC] + sz + [len(lags)], dtype=torch.float32).to(device)

    for i,lag in enumerate(lags):
        # print('Computing STA for lag %d' %lag)
        ix = inds[inds < NT-lag]      
        if batchsize is not None:
            for b in range(0, len(ix), batchsize):
                stim_ = modifier(stim[ix[b:b+batchsize],...])
                robs_ = robs[ix[b:b+batchsize]+lag,:]
                if dfs is not None:
                    dfs_ = dfs[ix[b:b+batchsize]+lag,:]
                    robs_ = robs_ * dfs_
                    div = dfs_.sum(dim=0)[:,None,None,None].to(device)
                else:
                    div = batchsize-lag
                sta[...,i] += torch.einsum('bchw, bn -> nchw', stim_.to(device),  robs_.to(device))/div
                torch.cuda.empty_cache()
        else:
            if dfs is not None:
                dfs_ = dfs[ix+lag,:]
                robs_ = robs[ix+lag,:]
                robs_ = robs_ * dfs_
                div = dfs_.sum(dim=0).to(device)
            else:
                robs_ = robs[ix+lag,:]
                div = NT-lag
            sta[...,i] = torch.einsum('bchw, bn -> nchw', modifier(stim[ix,...]).to(device),  robs_.to(device))/div

    return sta

def minmax(x, eps=1e-5):
    '''
    Normalize to be between 0 and 1
    TODO: add support for dimensions
    '''
    return (x-x.min().clamp(eps))/(x.max()-x.min().clamp(eps)).abs()

def center_of_mass(X, power=1, thresh = 0.1):
    '''
    Compute the center of mass of a 1D, 2D, or 3D array
    Input:
    X: 1D, 2D, or 3D tensor
    power: exponent to raise the values to before computing the center of mass (effectively a softmax)
    thresh: threshold to apply to the values before computing the center of mass
            this removes noise from the center of mass calculation
    Output:
        center of mass

    '''
    w = minmax(X)
    w[w < thresh] = 0.0
    if isinstance(w, torch.Tensor):
        w = w.detach().cpu().numpy()

    if len(w.shape) == 1:
        xx = np.arange(len(X))
        return np.sum(xx*w**power)/np.sum(w**power)
    elif len(w.shape) == 2:
        # 2D center of mass
        xx,yy = np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]))
        return np.sum(xx*w**power)/np.sum(w**power), np.sum(yy*w**power)/np.sum(w**power)
    elif len(w.shape) == 3:
        # 3D center of mass
        xx,yy,zz = np.meshgrid(np.arange(w.shape[1]), np.arange(w.shape[0]), np.arange(w.shape[2]))
        return np.sum(xx*w**power)/np.sum(w**power), np.sum(yy*w**power)/np.sum(w**power), np.sum(zz*w**power)/np.sum(w**power)
    

def get_spatial_power(spatial_power, sm=0, thresh=.5, pow=1):
    '''
    This function computes the spatial power of the STAs and returns the center

    Inputs:
        spower - torch.tensor of size (width, height, num_neurons) containing the spatial power of the STAs
        sm - integer specifying the size of the spatial window to use for computing the spatial power
    Outputs:
        spatial_power - torch.tensor of size (width, height, num_neurons) containing the spatial power of the STAs  
        ctr_x - float specifying the x-coordinate of the center of mass of the spatial power
        ctr_y - float specifying the y-coordinate of the center of mass of the spatial power

    '''
    
    
    if sm > 1:
        # smooth the spatial power
        spatial_power = torch.conv2d(spatial_power.unsqueeze(0).permute(3,0,1,2), torch.ones((1,1, sm,sm)), stride=1, padding='same')[:,0,:,:].permute(1,2,0)

    spatial_power = spatial_power**pow
    spatial_power[spatial_power < thresh*spatial_power.max()] = 0 # remove noise floor
    spatial_power /= spatial_power.sum()

    # xx,yy = torch.meshgrid(torch.linspace(-1, 1, spatial_power.shape[0]), torch.linspace(-1, 1, spatial_power.shape[1]))
    sz = spatial_power.shape
    xx,yy = torch.meshgrid(sz[0]*torch.linspace(-1/2, 1/2, sz[0]), sz[1]*torch.linspace(-1/2, 1/2, sz[1]))

    ctr_x = (spatial_power * yy).sum().item()
    ctr_y = (spatial_power * xx).sum().item()

    return spatial_power, ctr_x, ctr_y

def plot_weights_3Dsliced(sta, cmap='coolwarm_r'):
    '''
    Input:
    sta [NC, C, H, W, L]
        plot the sta when C = 1, H > 1, W > 1, L > 1
    '''

    NC = sta.shape[0]
    C = sta.shape[1]
    H = sta.shape[2]
    W = sta.shape[3]
    L = sta.shape[4]
    assert C == 1, "C must be 1"
    assert H > 1, "H must be > 1"
    assert W > 1, "W must be > 1"
    assert L > 1, "L must be > 1"

    # get subplot dimensions
    sx = int(np.ceil(np.sqrt(NC*2)))
    sy = int(np.round(np.sqrt(NC*2)))
    mod2 = sy % 2
    sy += mod2
    sx -= mod2

    fig = plt.figure(figsize=(sy, sx))
    for cc in range(NC):
        plt.subplot(sx,sy,cc+1)
        sta_ = sta[cc,...].mean(dim=0)
        hcom, wcom, tcom = torch.where(sta_.abs()==sta_.abs().max())
        tcom = tcom[0].item()
        # hcom,wcom,tcom = center_of_mass( (sta_-sta_.mean()).pow(2), power=3, thresh=.9)

        plt.subplot(sx,sy, cc*2 + 1)
        
        wspace = sta_[:,:,int(tcom)]
        vmax = wspace.abs().max().item()
        plt.imshow(wspace, aspect='auto', interpolation=None, cmap=cmap, extent=(-1,1,-1,1), vmin=-vmax, vmax=vmax)
        plt.axis('off')

        plt.subplot(sx,sy, cc*2 + 2)
        i,j=np.where(wspace==wspace.max().item())
        t1 = sta_[i[0], j[0],:]
        plt.plot(t1, '-b')
        
        # print(np.min(wspace.flatten()))
        i,j=np.where(wspace==wspace.min().item())
        t2 = sta_[i[0], j[0],:]
        plt.plot(t2, '-r')
        plt.axhline(0, color='k')
        plt.axvline(tcom, color='gray')

        plt.axis('off')
        plt.title(cc)
    
    return fig

def plot_weights_2D(sta, aspect='auto', interpolation=None, cmap="coolwarm_r"):

    '''
    Input:
    sta [NC, C, H, W, L]
        plot the sta when C = 1, H > 1, W > 1, L > 1
    '''

    NC = sta.shape[0]
    C = sta.shape[1]
    H = sta.shape[2]
    W = sta.shape[3]
    L = sta.shape[4]
    assert C == 1, "C must be 1"
    assert H > 1, "H must be > 1"
    assert W > 1, "W must be > 1"
    assert L == 1, "L must be 1"

    # get subplot dimensions
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    aspect_ratio = H/W
    if aspect_ratio > sx:
        sx = NC
        sy = 1

    # open figure with aspect ratio of the stimulus
    fig = plt.figure(figsize=(sx*2, sy*aspect_ratio*2))
    
    for cc in range(NC):
        plt.subplot(sy,sx,cc+1)
        sta_ = sta[cc,0,:,:,0]
        sta_ -= sta_.mean()
        vmax = sta_.abs().max().item()
        
        plt.imshow(sta_.detach().cpu().numpy(), aspect=aspect, interpolation=interpolation, cmap=cmap, vmax=vmax, vmin=-vmax)
        plt.axis('off')

        plt.title(cc)
    
    return fig

def plot_weights(sta, cmap='coolwarm_r'):
    '''
    plot weights for a given sta tensor
    Input:
    sta [NC, C, H, W, L]
    
    '''
    if len(sta.shape) == 4:
        sta = sta.unsqueeze(-1) # add a lag dimension if missing

    NC = sta.shape[0]
    C = sta.shape[1]
    H = sta.shape[2]
    W = sta.shape[3]
    L = sta.shape[4]

    ### sort through options and plot accordingly
    # Options are:
    # 1. C = 1, L = 1, W > 1, H > 1  plot as image for each cell
    # 2. C = 1, L > 1, W > 1, H = 1 plot as a space time image for each cell
    # 3. C = 1, L > 1, W = 1, H = 1 plot as a time series
    # 4. C = 1, L > 1, W > 1, H > 1 plot space at peak lag and time series at the peak and trough
    # 5. C > 1, L = 1, W > 1, H > 1 plot as image for each cell
    if C > 1:
        # collapse channels into one spatial dimension
        sta = sta.reshape([NC, 1, C*H, W, L])
        aspect='equal'
        C = 1
    else:
        aspect='auto'

    if C == 1 and L == 1 and W > 1 and H > 1:
        fig = plot_weights_2D(sta, aspect=aspect, cmap=cmap)

    elif C == 1 and L > 1 and W > 1 and H > 1:
        fig = plot_weights_3Dsliced(sta, cmap=cmap)

    else:
        raise NotImplementedError

    return fig


def get_mask_from_contour(Im, contour):
    import scipy.ndimage as ndimage    
    # Create an empty image to store the masked array
    r_mask = np.zeros_like(Im, dtype='bool')
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    # Fill in the hole created by the contour boundary
    r_mask = ndimage.binary_fill_holes(r_mask)
    return r_mask

def get_contour(Im, thresh):
    # use skimage to find contours at a threhsold,
    # select the largest contour and return the area and center of mass
    
    # find contours in Im at threshold thresh
    contours = measure.find_contours(Im, thresh)
    # Select the largest contiguous contour
    contour = sorted(contours, key=lambda x: len(x))[-1]
    
    r_mask = get_mask_from_contour(Im, contour)
    # plt.imshow(r_mask, interpolation='nearest', cmap=plt.cm.gray)
    # plt.show()

    M = measure.moments(r_mask, 1)

    area = M[0, 0]
    center = np.asarray([M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]])
    
    return contour, area, center


def get_rf_contour(rf, thresh = .5):
    '''

    '''
    
    thresh_ = .9

    assert thresh < 1, 'get_rf_contour: threshold must be 0 < thresh < 1. use get_contour for unnormalized values'

    con0, _, _ = get_contour(rf, thresh=thresh_)

    cond = True
    while cond:

        thresh_ = thresh_ - .1
        _, _, ctr = get_contour(rf, thresh=thresh_)
        inpoly = measure.points_in_poly(ctr[None,:], con0)[0]
        if inpoly and thresh_ >= thresh:
            continue
        else:
            thresh_ = thresh_ + .1
            cond = False

    con, ar, ctr = get_contour(rf, thresh=thresh_)
    
    return con, ar, ctr, thresh_

def shift_stim_fourier(s, shift, size=None, batch_size=None):
    # apply a shift to each channel of x using the fourier shift theorem
    # x: (n c, x, y)
    # shift: (n, 2)
    # returns: (n, c, x, y)
    n, c, x, y = s.shape
    X, Y = torch.meshgrid(
        torch.fft.fftfreq(x, 1 / x),
        torch.fft.rfftfreq(y, 1 / y),
        indexing='ij'
    )

    X = torch.exp(-2 * torch.pi * 1j * X[None,:,:] * -shift[:, 1].unsqueeze(1).unsqueeze(1) / x).unsqueeze(1)
    Y = torch.exp(-2 * torch.pi * 1j * Y[None,:,:] * -shift[:, 0].unsqueeze(1).unsqueeze(1) / y).unsqueeze(1)

    fft = torch.fft.rfft2(s)
    fft_shifted = fft * X * Y
    
    #create hanning window
    wx = torch.fft.fftshift(torch.hann_window(x, periodic=False))
    wy = torch.hann_window(y, periodic=False)[-(1 + y//2):]
    w = (wx[:, None] * wy[None, :])**0.5
    # print(fft_shifted.shape, w.shape, X.s/hape, Y.shape, wy.shape, wx.shape)
    fft_shifted = fft_shifted * w
    
    shifted = torch.fft.irfft2(fft_shifted).real
    
    if size is not None:
        # croph = (y-size[1])//2
        # cropw = (x-size[0])//2
        # shifted = shifted[:,:,croph:-croph-1,cropw:-cropw-1]
        start_y = (y - size[1])//2
        end_y = start_y + size[1] 
        start_x = (x - size[0]) // 2
        end_x = start_x + size[0]
        shifted = shifted[:, :, start_y:end_y, start_x:end_x]
    
    # if batch_size is not None:
    #     # raise NotImplementedError
    #     raise NotImplementedError, "batch_size not implemented for fourier shifter"
        
    return shifted

    
def shift_stim(stim, shift, size, scale = 1, grid_scale=1, mode='bilinear', batch_size=None, upsample_factor=1, upsample_mode='nearest', verbose=False, no_grad=False):
    '''
    This function samples a grid of size (size[0], size[1]) from the stimulus at the locations
    specified by shift. The shift is in pixels and is a 2D vector. The output is a tensor of size
    (stim.shape[0], stim.shape[1], size[0], size[1]) and the grid is returned as a tensor of size
    (stim.shape[0], size[0], size[1], 2) so that it can be used with grid_sample (or diplayed for
    animations that we should make to demo how this all works)

    Inputs:
        stim: torch.tensor of size (stim.shape[0], stim.shape[1], stim.shape[2], stim.shape[3]) 
            where the first dimension is the batch dimension
        shift: torch.tensor of size (stim.shape[0], 2) where the second dimension is the x and y
            shift in pixels
        size: list of length 2 specifying the size of the output grid (in pixels)
        mode: string specifying the interpolation mode for grid_sample (default is bilinear)
    
    Outputs:
        Image: torch.tensor of size (stim.shape[0], stim.shape[1], size[0], size[1]) where the first
            dimension is the batch dimension
        Grid: torch.tensor of size (stim.shape[0], size[0], size[1], 2) where the last dimension is
            the x and y coordinates of the grid
    '''
    
    import torch.nn.functional as F
    from tqdm import tqdm
    import gc

    
    dy, dx = torch.meshgrid(torch.arange(size[0])-size[0]/2, torch.arange(size[1])-size[1]/2, indexing='ij')
    scalex = (stim.shape[2]/2*scale)
    scaley = (stim.shape[3]/2*scale)
    dx = dx / scalex * grid_scale
    dy = dy / scaley * grid_scale
    dx = dx.unsqueeze(0).unsqueeze(-1).to(stim.device)
    dy = dy.unsqueeze(0).unsqueeze(-1).to(stim.device)
    if verbose:
        print("Building grid...")
        print("grid size: (dx: %d, dy: %d)" % (dx.shape[2], dy.shape[1]))

    sx = shift[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/scalex
    sy = shift[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/scaley

    if verbose:
        print("shift size: (sx: %d, sy: %d)" % (sx.shape[2], sy.shape[1]))

    grid = torch.cat((dx+sx, dy+sy), dim=-1).to(stim.device) # TODO: make this batched if requested 
    del dx, dy, sx, sy
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print("shifting and resampling...")
        if batch_size is not None:
            loop = tqdm(range(0, stim.shape[0], batch_size))
    else:
        if batch_size is not None:
            loop = range(0, stim.shape[0], batch_size)

    if batch_size is not None:
        out = torch.zeros((stim.shape[0], stim.shape[1], size[0], size[1]), device=stim.device)
        for b in loop:
            stim_batch = stim[b:b+batch_size]
            if upsample_factor > 1:
                if no_grad:
                    with torch.no_grad():
                        stim_batch = F.interpolate(stim_batch, scale_factor=upsample_factor, mode=upsample_mode)
                else:
                    stim_batch = F.interpolate(stim_batch, scale_factor=upsample_factor, mode=upsample_mode)

            if no_grad:
                with torch.no_grad():
                    out[b:b+batch_size] = F.grid_sample(stim_batch, grid[b:b+batch_size], mode=mode, align_corners=True)
            else:
                out[b:b+batch_size] = F.grid_sample(stim_batch, grid[b:b+batch_size], mode=mode, align_corners=True)

            gc.collect()
    else:
        if upsample_factor > 1:
            stim = F.interpolate(stim, scale_factor=upsample_factor, mode=upsample_mode)
        out = F.grid_sample(stim, grid, mode=mode, align_corners=True)

    return out, grid