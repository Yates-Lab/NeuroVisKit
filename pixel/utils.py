import os
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils0 import ss, ensure_dir, reporthook

""" Available Datasets:
1. FixationMultiDataset - generates fixations from multiple stimulus classes 
                        and experimental sessions. no time-embedding
2. PixelDataset - time-embedded 2D free-viewing movies and spike trains
"""
def get_stim_list(id=None, verbose=False):

    stim_list = {
            '20191119': 'logan_20191119_-20_-10_50_60_0_19_0_1.hdf5',
            '20191119C': 'logan_20191119_-40__7_40_87_1_1_1_19_0_0.hdf5',
            '20191120a':'logan_20191120a_-20_-10_50_60_0_19_0_1.hdf5',
            '20191120C': 'logan_20191120a_-35_-5_45_75_1_1_1_19_0_0.hdf5',
            '20191121': 'logan_20191121_-20_-20_50_50_0_19_0_1.hdf5',
            '20191121C': 'logan_20191121_-26_-26_54_54_1_1_1_19_0_0.hdf5',
            '20191122': 'logan_20191122_-20_-10_50_60_0_19_0_1.hdf5',
            '20191202C': 'logan_20191202_-38_-4_42_76_1_1_1_19_0_0.hdf5',
            '20191205': 'logan_20191205_-20_-10_50_60_0_19_0_1.hdf5',
            '20191206': 'logan_20191206_-20_-10_50_60_0_19_0_1.hdf5',
            '20191210C': 'logan_20191210_-24_-5_56_75_1_1_1_19_0_0.hdf5',
            '20191220C': 'logan_20191220_-30_-12_50_68_1_1_1_19_0_0.hdf5',
            '20191223C': 'logan_20191223_-37_-11_43_69_1_1_1_19_0_0.hdf5',
            '20191224C': 'logan_20191224_-78_-115___2_-35_1_1_1_19_0_0.hdf5',
            '20191226C': 'logan_20191226_-31_-17_49_63_1_1_1_19_0_0.hdf5',
            '20191227C': 'logan_20191227_-50_-15_30_65_1_1_1_19_0_0.hdf5',
            '20191230C': 'logan_20191230_-38_-13_42_67_1_1_1_19_0_0.hdf5',
            '20191231': 'logan_20191231_-20_-10_50_60_0_19_0_1.hdf5',
            '20191231C': 'logan_20191231_-36_-7_44_73_1_1_1_19_0_0.hdf5',
            '20200106C': 'logan_20200106_-37_-11_43_69_1_1_1_19_0_0.hdf5',
            '20200107C': 'logan_20200107_-40_-12_40_68_1_1_1_19_0_0.hdf5',
            '20200109C': 'logan_20200109_-33_-12_47_68_1_1_1_19_0_0.hdf5',
            '20200110C': 'logan_20200110_-36_-12_44_68_1_1_1_19_0_0.hdf5',
            '20200115C': 'logan_20200115_-40_-7_40_73_1_1_1_19_0_0.hdf5',
            '20200228C': 'logan_20200228_-31_-2_49_78_1_1_1_19_0_0.hdf5',
            '20200229C': 'logan_20200229_-31_-21_49_59_1_1_1_19_0_0.hdf5',
            '20200302C': 'logan_20200302_-31_-12_49_68_1_1_1_19_0_0.hdf5',
            '20200303C': 'logan_20200303_-35_-12_45_68_1_1_1_19_0_0.hdf5',
            '20200304': 'logan_20200304_-20_-10_50_60_0_19_0_1.hdf5',
            '20200304B': 'logan_20200304_-66_-37__84_113_1_0_1_41_0_0_1.hdf5',
            '20200304C': 'logan_20200304_-31_-2_49_78_1_1_1_19_0_0.hdf5',
            '20200306C': 'logan_20200306_-45_-12_35_68_1_1_1_19_0_0.hdf5',
            '20220216': 'allen_20220216_-60_-50_10_20_0_19_0_1.hdf5',
            '20220601': 'allen_20220601_-80_-50_10_50_1_0_1_19_0_1.hdf5',
            '20220601C': 'allen_20220601_-76_-29__4_51_1_1_1_19_0_0.hdf5',
            '20220601fix': 'allen_20220601_-80_-50_10_50_0_0_0_19_0_1.hdf5',
            '20220601fixc': 'allen_20220601_-80_-50_10_50_1_0_0_19_0_1.hdf5',
            '20220610': 'allen_20220610_-80_-50_10_50_1_0_1_19_0_1.hdf5',
            '20220610C': 'allen_20220610_-76_-39__4_41_1_1_1_19_0_0.hdf5',
            '20220610fix': 'allen_20220610_-80_-50_10_50_0_0_0_19_0_1.hdf5',
            '20220610fixc': 'allen_20220610_-80_-50_10_50_1_0_0_19_0_1.hdf5',
            '20220805C': 'allen_20220805_-78_-33__2_47_1_1_1_19_0_0.hdf5',
            '20230608': 'r20230618_-50___0_200_250_1_0_1_3_0_0.hdf5'
        }
    


    if id is None:
        for str in list(stim_list.keys()):
            if verbose:
                print(str)
        return stim_list

    if id not in stim_list.keys():
        raise ValueError('Stimulus not found')
    
    return stim_list[id]


def get_stim_url(id):
    urlpath = {
            '20191119': 'https://www.dropbox.com/s/xxaat202j20kriy/logan_20191119_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20191231':'https://www.dropbox.com/s/ulpcjfb48c6dyyf/logan_20191231_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20191231C': 'https://www.dropbox.com/s/dmzueruyc9iruya/logan_20191231_-36_-7_44_73_1_1_1_19_0_0.hdf5?dl=1',
            '20200304': 'https://www.dropbox.com/s/5tj5m2rp0wht8z2/logan_20200304_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20200304C': 'https://www.dropbox.com/s/gwjcqzh7muh2q7q/logan_20200304_-31_-2_49_78_1_1_1_19_0_0.hdf5?dl=1',
            '20220601': 'https://www.dropbox.com/s/v85m0b9kzwiowqm/allen_20220601_-80_-50_10_50_1_0_19_0_1.hdf5?dl=1',
            '20220610': 'https://www.dropbox.com/s/0tkwqlfjdarkrqm/allen_20220610_-80_-50_10_50_1_0_1_19_0_1.hdf5?dl=1',
            '20200304B': 'https://www.dropbox.com/scl/fi/278v990md2oriu8u4km33/logan_20200304_-66_-37__84_113_1_0_1_41_0_0_1.hdf5?rlkey=ws5480sdfcgc10b0m4w07umwg&dl=1'
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]

def get_shifter_url(id):
    urlpath = {
            '20191119': 'https://www.dropbox.com/s/dd05gxt8l8hmw3o/shifter_20191119_kilowf.p?dl=1',
            '20191120a': 'https://www.dropbox.com/s/h4elcp46le5tet0/shifter_20191120a_kilowf.p?dl=1',
            '20191121': 'https://www.dropbox.com/s/rfzefex8diu5ts5/shifter_20191121_kilowf.p?dl=1',
            '20191122': 'https://www.dropbox.com/s/2me7yauvpprnv0b/shifter_20191122_kilowf.p?dl=1',
            '20191205': 'https://www.dropbox.com/s/r56wt4rfozmjiy8/shifter_20191205_kilowf.p?dl=1',
            '20191206': 'https://www.dropbox.com/s/qec8cats077bx8c/shifter_20191206_kilowf.p?dl=1',
            '20200304': 'https://www.dropbox.com/s/t0j8k55a8jexgt4/shifter_20210304_kilowf.p?dl=1',
            '20220610': 'https://www.dropbox.com/s/3ilpht9jw0insvu/shifter_20220610_kilo.p?dl=1',
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]


def download_set(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_stim_url(sessname)
    fout = os.path.join(fpath, get_stim_list(sessname))
    print("Downloading...") 
    import urllib
    urllib.request.urlretrieve(url, fout, reporthook)
    print("Done")

def download_shifter(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_shifter_url(sessname)
    fout = os.path.join(fpath, 'shifter_' + sessname + '_kilowf.p')
    print("Downloading...") 
    import urllib
    urllib.request.urlretrieve(url, fout, reporthook)
    print("Done")


def shift_im(stim, shift, affine=False, mode='nearest'):
    '''
    Primary function for shifting the intput stimulus
    Inputs:
        stim [Batch x channels x height x width] (use Fold2d to fold lags if necessary)
        shift [Batch x 2] or [Batch x 4] if translation only or affine
        affine [Boolean] set to True if using affine transformation
        mode [str] 'nearest' (default)
    '''
    import torch.nn.functional as F
    batch_size = stim.shape[0]

    # build affine transformation matrix size = [batch x 2 x 3]
    affine_trans = torch.zeros((batch_size, 2, 3) , dtype=stim.dtype, device=stim.device)
    
    if affine:
        # fill in rotation and scaling
        costheta = torch.cos(shift[:,2].clamp(-.5, .5))
        sintheta = torch.sin(shift[:,2].clamp(-.5, .5))
        
        scale = shift[:,3]**2 + 1.0

        affine_trans[:,0,0] = costheta*scale
        affine_trans[:,0,1] = -sintheta*scale
        affine_trans[:,1,0] = sintheta*scale
        affine_trans[:,1,1] = costheta*scale

    else:
        # no rotation or scaling
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

    # translation
    affine_trans[:,0,2] = shift[:,0]
    affine_trans[:,1,2] = shift[:,1]

    grid = F.affine_grid(affine_trans, stim.shape, align_corners=False)

    return F.grid_sample(stim, grid, mode=mode, align_corners=False).detach()


def plot_shifter(shifter, valid_eye_rad=5.2, ngrid = 100, title=None, show=True):
        import matplotlib.pyplot as plt
        xx,yy = np.meshgrid(np.linspace(-valid_eye_rad, valid_eye_rad,ngrid),np.linspace(-valid_eye_rad, valid_eye_rad,ngrid))
        xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
        ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

        inputs = torch.cat( (xgrid,ygrid), dim=1)

        xyshift = shifter(inputs).detach().numpy()

        # xyshift/=valid_eye_rad/60 # conver to arcmin
        vmin = np.min(xyshift)
        vmax = np.max(xyshift)

        nshift = xyshift.shape[1]
        shift = []
        fig = plt.figure(figsize=(3*nshift,3))
        for ishift in range(nshift):
            shift.append(xyshift[:,ishift].reshape((ngrid, ngrid)))
            plt.subplot(1,nshift, ishift+1)
            plt.imshow(shift[-1], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None)
            plt.colorbar()
        
        # # shift = [xyshift[:,0].reshape((ngrid,ngrid))]
        # # shift.append(xyshift[:,1].reshape((ngrid,ngrid))) 
        
        # plt.subplot(1,2,1)
        # plt.imshow(shift[0], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        # plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.imshow(shift[1], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        if title is not None:
            plt.suptitle(title)

        if show:
            plt.show()
        
        return shift, fig
    
def firingrate_datafilter( fr, Lmedian=10, Lhole=30, FRcut=1.0, frac_reject=0.1, to_plot=False, verbose=False ):
    """Generate data filter for neuron given firing rate over time"""
    if to_plot:
        verbose = True
    def median_smoothing( f, L=5):
        mout = deepcopy(f)
        for tt in range(L, len(f)-L):
            mout[tt] = np.median(f[np.arange(tt-L,tt+L)])
        return mout
    mx = median_smoothing(fr, L=Lmedian)
    df = np.zeros(len(mx))
    # pre-filter ends
    m = np.median(fr)
    v = np.where((mx >= m-np.sqrt(m)) & (mx <= m+np.sqrt(m)))[0]
    df[range(v[0], v[-1])] = 1
    m = np.median(fr[df > 0])
    if m < FRcut:
        # See if can salvage: see if median of 1/4 of data is above FRcut
        msplurge = np.zeros(4)
        L = len(mx)//4
        for ii in range(4):
            msplurge[ii] = np.median(mx[range(L*ii, L*(ii+1))])
        m = np.max(msplurge)
        if m < FRcut:
            if verbose:
                print('  Median criterium fail: %f'%m)
            return np.zeros(len(mx))
        # Otherwise back in game: looking for higher median
    v = np.where((mx >= m-np.sqrt(m)) & (mx <= m+np.sqrt(m)))[0]
    df = np.zeros(len(mx))
    df[range(v[0], v[-1]+1)] = 1
    # Last
    m = np.median(fr[df > 0])
    v = np.where((mx >= m-np.sqrt(m)) & (mx <= m+np.sqrt(m)))[0]
    # Look for largest holes
    ind = np.argmax(np.diff(v))
    largest_hole = np.arange(v[ind], v[ind+1])
    if len(largest_hole) > Lhole:
        if verbose:
            print('  Removing hole size=%d'%len(largest_hole))
        df[largest_hole] = 0
        # Decide whether to remove one side of hole or other based on consistent firing rate change after the hole
        chunks = [np.arange(v[0], largest_hole[0]), np.arange(largest_hole[-1], v[-1])]
        mfrs = [np.median(fr[chunks[0]]), np.median(fr[chunks[1]])]

        if (len(chunks[0]) > len(chunks[1])) & (mfrs[0] > FRcut):
            dom = 0
        else: 
            dom = 1
        if ((mfrs[dom] > mfrs[1-dom]) & (mfrs[1-dom] < mfrs[dom]-np.sqrt(mfrs[dom]))) | \
                ((mfrs[dom] < mfrs[1-dom]) & (mfrs[1-dom] > mfrs[dom]+np.sqrt(mfrs[dom]))):
            #print('eliminating', 1-dom)
            df[chunks[1-dom]] = 0

    # Eliminate any small islands of validity (less than Lhole)
    a = np.where(df == 0)[0]  # where there are zeros
    if len(a) > 0:
        b = np.diff(a) # the length of islands (subtracting 1: 0 is no island)
        c = np.where((b > 1) & (b < Lhole))[0]  # the index of the islands that are the wrong size
        # a[c] is the location of the invalid islands, b[c] is the size of these islands (but first point is zero before) 
        for ii in range(len(c)):
            df[a[c[ii]]+np.arange(b[c[ii]])] = 0

    # Final reject criteria: large fraction of mean firing rates in trial well above median poisson limit
    m = np.median(fr[df > 0])
    #stability_ratio = len(np.where(Ntrack[df > 0,cc] > m+np.sqrt(m))[0])/np.sum(df > 0)
    stability_ratio = len(np.where(abs(mx[df > 0]-m) > np.sqrt(m))[0])/np.sum(df > 0)
    if stability_ratio > frac_reject:
        if verbose:
            print('  Stability criteria not met:', stability_ratio)
        df[:] = 0
    if to_plot:
        ss(2,1)
        plt.subplot(211)
        plt.plot(fr,'b')
        plt.plot(mx,'g')
        plt.plot([0,len(fr)], [m, m],'k')
        plt.plot([0,len(fr)], [m-np.sqrt(m), m-np.sqrt(m)],'r')
        plt.plot([0,len(fr)], [m+np.sqrt(m), m+np.sqrt(m)],'r')
        plt.subplot(212)
        plt.plot(df)
        plt.show()

    return df