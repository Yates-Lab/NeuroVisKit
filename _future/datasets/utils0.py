import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

def subplot_setup(num_rows, num_cols, row_height=2, fighandle=False):
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)
    fig.tight_layout()
    if fighandle is True:
        return fig


def ss( num_rows=1, num_cols=1, row_height=2.5, rh=None, fighandle=False):
    # Short-hand for subplot_setup with useful defaults for quick usage
    if rh is not None:
        row_height=rh  # really to just save a bit of typing
    subplot_setup(num_rows, num_cols, row_height=row_height, fighandle=fighandle)

def is_int( val ):
    """returns Boolean as to whether val is one of many types of integers"""
    if isinstance(val, int) or \
        isinstance(val, np.int) or isinstance(val, np.int32) or isinstance(val, np.int64) or \
        (isinstance(val, np.ndarray) and (len(val.shape) == 0)):
        return True
    else:
        return False

def download_file(url: str, file_name: str):
    '''
    Downloads file from url and saves it to file_name.
    '''
    import urllib.request
    print("Downloading %s to %s" % (url, file_name))
    urllib.request.urlretrieve(url, file_name, reporthook)
    
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def ensure_dir(dir_name: str):
    '''
    Creates folder if not exists.
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def downsample_time(x, ds, flipped=None):
    
    NTold = x.shape[0]
    dims = x.shape[1]
    
    if flipped is None:
        flipped = False
        if dims > NTold:
            # then assume flipped
            flipped = True
            x = x.T
    
    NTnew = np.floor(NTold/ds).astype(int)
    if type(x) is torch.Tensor:
        y = torch.zeros((NTnew, dims), dtype=x.dtype)
    else:
        y = np.zeros((NTnew, dims))
        
    for nn in range(ds-1):
        y[:,:] = y[:,:] + x[nn + np.arange(0, NTnew, 1)*ds,:]
    
    if flipped:
        y = y.T
        
    return y

def resample_time(data, torig, tout):
    ''' resample data at times torig at times tout '''
    ''' data is components x time '''
    ''' credit to Carsen Stringer '''
    fs = torig.size / tout.size # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs/4), axis=1)
    f = interp1d(torig, data, kind='linear', axis=0, fill_value='extrapolate')
    dout = f(tout)
    return dout

def bin_population(times, clu, btimes, cids,
        maxbsize=1, padding=0, dtype=torch.float32):
    ''' bin time points (times) at btimes'''
    NC = np.max(cids) + 1
    robs = torch.zeros((len(btimes), NC), dtype=dtype)
    inds = np.argsort(btimes)
    ft = btimes[inds]
    for cc in range(NC):
        cnt = bin_at_frames(times[clu==cids[cc]], ft, maxbsize=maxbsize, padding=padding)
        robs[inds,cc] = torch.tensor(cnt, dtype=dtype)
    return robs

def bin_population_sparse(times, clu, btimes, cids, dtype=torch.float32, to_dense=True):
    ''' bin time points (times) at btimes'''
    NC = np.max(cids)+1
    robs = torch.sparse_coo_tensor( (np.digitize(times, btimes)-1, clu), np.ones(len(clu)), (len(btimes), NC) , dtype=dtype)
    if to_dense:
        return robs.to_dense()
    else:
        return robs

def bin_at_frames(times, btimes, maxbsize=1, padding=0):
    ''' bin time points (times) at btimes'''
    breaks = np.where(np.diff(btimes)>maxbsize)[0]
    
    # add extra bin edge
    btimes = np.append(btimes, btimes[-1]+maxbsize)

    out,_ = np.histogram(times, bins=btimes)
    out = out.astype('float32')

    if padding > 0:
        out2 = out[range(breaks[0])]
        dt = np.median(np.diff(btimes))
        pad = np.arange(1,padding+1, 1)*dt
        for i in range(1,len(breaks)):
            tmp,_ = np.histogram(times, pad+btimes[breaks[i]])
            out2.append(tmp)
            out2.append(out[range(breaks[i-1]+1, breaks[i])])            
    else:
        out[breaks] = 0.0
    
    return out

def r_squared(true, pred, data_indxs=None):
    """
    START.

    :param true: vector containing true values
    :param pred: vector containing predicted (modeled) values
    :param data_indxs: obv.
    :return: R^2

    It is assumed that vectors are organized in columns

    END.
    """

    assert true.shape == pred.shape, 'true and prediction vectors should have the same shape'

    if data_indxs is None:
        dim = true.shape[0]
        data_indxs = np.arange(dim)
    else:
        dim = len(data_indxs)

    ss_res = np.sum(np.square(true[data_indxs, :] - pred[data_indxs, :]), axis=0) / dim
    ss_tot = np.var(true[data_indxs, :], axis=0)

    return 1 - ss_res/ss_tot


def create_time_embedding(stim, pdims, up_fac=1, tent_spacing=1):
    """All the arguments starting with a p are part of params structure which I 
    will fix later.
    
    Takes a Txd stimulus matrix and creates a time-embedded matrix of size 
    Tx(d*L), where L is the desired number of time lags. If stim is a 3d array, 
    the spatial dimensions are folded into the 2nd dimension. 
    
    Assumes zero-padding.
     
    Optional up-sampling of stimulus and tent-basis representation for filter 
    estimation.
    
    Note that xmatrix is formatted so that adjacent time lags are adjacent 
    within a time-slice of the xmatrix, thus x(t, 1:nLags) gives all time lags 
    of the first spatial pixel at time t.
    
    Args:
        stim (type): simulus matrix (time must be in the first dim).
        pdims (list/array): length(3) list of stimulus dimensions
        up_fac (type): description
        tent_spacing (type): description
        
    Returns:
        numpy array: time-embedded stim matrix
        
    """
    
    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))
        print('Flattening stimulus to produce design matrix.')
    elif len(sz) == 1:
        stim = np.expand_dims(stim, axis=1)
    sz = list(np.shape(stim))

    # No support for more than two spatial dimensions
    if len(sz) > 3:
        print('More than two spatial dimensions not supported, but creating xmatrix anyways...')

    # Check that the size of stim matches with the specified stim_params
    # structure
    if np.prod(pdims[1:]) != sz[1]:
        print('Stimulus dimension mismatch')
        raise ValueError

    modstim = deepcopy(stim)
    # Up-sample stimulus if required
    if up_fac > 1:
        # Repeats the stimulus along the time dimension
        modstim = np.repeat(modstim, up_fac, 0)
        # Since we have a new value for time dimension
        sz = list(np.shape(modstim))

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append(
            np.arange(1, tent_spacing) / tent_spacing,
            1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + \
                            shift_mat_zpad(modstim,
                                           ii-tent_spacing+1,
                                           0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing

    # If tent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using
    # tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).
    # Since isempty(tent_spa...) is equivalent to its value being 1 I added
    # this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])),
                        np.concatenate((modstim[0], np.zeros(pdims[0] - 1)),
                                       axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros((sz[0], np.prod(pdims)))
        for lag in range(pdims[0]):
            for xx in range(0, sz[1]):
                xmat[:, xx*pdims[0]+lag] = shift_mat_zpad(
                    modstim[:, xx], lag_spacing * lag, 0)

    return xmat
# END create_time_embedding


def design_matrix_tent_basis( s, anchors, zero_left=False, zero_right=False):
    """Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
    Here s is a continuous variable (e.g., a stimulus) that is function of time -- single dimension --
    and this will generate apply a tent basis set to s with a basis variable for each anchor point. 
    The end anchor points will be one-sided, but these can be dropped by changing "zero_left" and/or
    "zero_right" into "True".

    Inputs: 
        s: continuous one-dimensional variable with NT time points
        anchors: list or array of anchor points for tent-basis set
        zero_left, zero_right: boolean whether to drop the edge bases (default for both is False)
    Outputs:
        X: design matrix that will be NT x the number of anchors left after zeroing out left and right
    """

    if len(s.shape) > 1:
        assert s.shape[1] == 1, 'Can only work on 1-d variables currently'
        s = np.squeeze(s)

    NT = len(s)
    NA = len(anchors)
    X = np.zeros([NT, NA])
    for nn in range(NA):
        if nn == 0:
            #X[np.where(s < anchors[0])[0], 0] = 1
            X[:, 0] = 1
        else:
            dx = anchors[nn]-anchors[nn-1]
            X[:, nn] = np.minimum(np.maximum(np.divide( deepcopy(s)-anchors[nn-1], dx ), 0), 1)
        if nn < NA-1:
            dx = anchors[nn+1]-anchors[nn]
            X[:, nn] *= np.maximum(np.minimum(np.divide(np.add(-deepcopy(s), anchors[nn+1]), dx), 1), 0)
    if zero_left:
        X = X[:,1:]
    if zero_right:
        X = X[:,:-1]
    return X


def tent_basis_generate( xs=None, num_params=None, doubling_time=None, init_spacing=1, first_lag=0 ):
    """Computes tent-bases over the range of 'xs', with center points at each value of 'xs'.
    Alternatively (if xs=None), will generate a list with init_space and doubling_time up to
    the total number of parameters. Must specify xs OR num_params. 
    Note this assumes discrete (binned) variables to be acted on.
    
    Defaults:
        doubling_time = num_params
        init_space = 1"""

    # Determine anchor-points
    if xs is not None:
        tbx = np.array(xs,dtype='int32')
        if num_params is not None: 
            print( 'Warning: will only use xs input -- num_params is ignored.' )
    else:
        assert num_params is not None, 'Need to specify either xs or num_params'
        if doubling_time is None:
            doubling_time = num_params+1  # never doubles
        tbx = np.zeros( num_params, dtype='int32' )
        cur_loc, cur_spacing, sp_count = first_lag, init_spacing, 0
        for nn in range(num_params):
            tbx[nn] = cur_loc
            cur_loc += cur_spacing
            sp_count += 1
            if sp_count == doubling_time:
                sp_count = 0
                cur_spacing *= 2

    # Generate tent-basis given anchor points
    NB = len(tbx)
    NX = (np.max(tbx)+1).astype(int)
    tent_basis = np.zeros([NX,NB], dtype='float32')
    for nn in range(NB):
        if nn > 0:
            dx = tbx[nn]-tbx[nn-1]
            tent_basis[range(tbx[nn-1], tbx[nn]+1), nn] = np.array(list(range(dx+1)))/dx
        elif tbx[0] > 0:  # option to have function go to zero at beginning
            dx = tbx[0]
            tent_basis[range(tbx[nn]+1), nn] = np.array(list(range(dx+1)))/dx
        if nn < NB-1:
            dx = tbx[nn+1]-tbx[nn]
            tent_basis[range(tbx[nn], tbx[nn+1]+1), nn] = 1-np.array(list(range(dx+1)))/dx

    return tent_basis


def shift_mat_zpad(x, shift, dim=0):
    """Takes a vector or matrix and shifts it along dimension dim by amount 
    shift using zero-padding. Positive shifts move the matrix right or down.
    
    Args:
        x (type): description
        shift (type): description
        dim (type): description
        
    Returns:
        type: description
            
    Raises:
            
    """

    assert x.ndim < 3, 'only works in 2 dims or less at the moment.'
    if x.ndim == 1:
        oneDarray = True
        xcopy = np.zeros([len(x), 1])
        xcopy[:, 0] = x
    else:
        xcopy = deepcopy(x)
        oneDarray = False
    sz = list(np.shape(xcopy))

    if sz[0] == 1:
        dim = 1

    if dim == 0:
        if shift >= 0:
            a = np.zeros((shift, sz[1]))
            b = xcopy[0:sz[0]-shift, :]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((-shift, sz[1]))
            b = xcopy[-shift:, :]
            xshifted = np.concatenate((b, a), axis=dim)
    elif dim == 1:
        if shift >= 0:
            a = np.zeros((sz[0], shift))
            b = xcopy[:, 0:sz[1]-shift]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((sz[0], -shift))
            b = xcopy[:, -shift:]
            xshifted = np.concatenate((b, a), axis=dim)

    # If the shift in one direction is bigger than the size of the stimulus in
    # that direction return a zero matrix
    if (dim == 0 and abs(shift) > sz[0]) or (dim == 1 and abs(shift) > sz[1]):
        xshifted = np.zeros(sz)

    # Make into single-dimension if it started that way
    if oneDarray:
        xshifted = xshifted[:,0]

    return xshifted
# END shift_mat_zpad

def generate_xv_folds(nt, num_folds=5, num_blocks=3, which_fold=None):
    """Will generate unique and cross-validation indices, but subsample in each block
        NT = number of time steps
        num_folds = fraction of data (1/fold) to set aside for cross-validation
        which_fold = which fraction of data to set aside for cross-validation (default: middle of each block)
        num_blocks = how many blocks to sample fold validation from"""

    test_inds = []
    NTblock = np.floor(nt/num_blocks).astype(int)
    block_sizes = np.zeros(num_blocks, dtype='int32')
    block_sizes[range(num_blocks-1)] = NTblock
    block_sizes[num_blocks-1] = nt-(num_blocks-1)*NTblock

    if which_fold is None:
        which_fold = num_folds//2
    else:
        assert which_fold < num_folds, 'Must choose XV fold within num_folds =' + str(num_folds)

    # Pick XV indices for each block
    cnt = 0
    for bb in range(num_blocks):
        tstart = np.floor(block_sizes[bb] * (which_fold / num_folds))
        if which_fold < num_folds-1:
            tstop = np.floor(block_sizes[bb] * ((which_fold+1) / num_folds))
        else: 
            tstop = block_sizes[bb]

        test_inds = test_inds + list(range(int(cnt+tstart), int(cnt+tstop)))
        cnt = cnt + block_sizes[bb]

    test_inds = np.array(test_inds, dtype='int')
    train_inds = np.setdiff1d(np.arange(0, nt, 1), test_inds)

    return train_inds, test_inds