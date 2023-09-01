import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import os
from datasets.utils import ensure_dir, reporthook, downsample_time
from .utils import get_stim_list, download_set

class PixelDataset(Dataset):
    """
    PixelDataset is a pytorch Dataset for loading stimulus movies and spikes
    Arguments:
    id:             <string>    id of the session (must exist in get_stim_list, e.g,. '20200304)
    num_lags:       <int>       number of time lags
    stims:          <list>      list of strings corresponding to requested stimuli
                                    "Gabor"     - gabor droplet noise
                                    "Grating"   - full-field gratings
                                    "BackImage" - static natural image 
                                    "FixRsvpStim" - rapidly flashed filtered natural images
    stimset:        <string>    "Train" or "Test" set
    downsample_s    <int>       spatial downsample factor (this slows things down, because smoothing before subsample)
    downsample_t    <int>       temporal downsample factor (this slows things down because of smoothing operation)
    valid_eye_rad   <float>     valid region on screen (centered on 0,0) in degrees of visual angle
    fixations_only  <bool>      whether to only include fixations
    dirname         <string>    full path to where the data are stored
    cids            <list>      list of cells to include (file is loaded in its entirety and then sampled because slicing hdf5 has to be simple)
    cropidx                     index of the form [(x0,x1),(y0,y1)] or None type
    include_eyepos  <bool>      flag to include the eye position info in __get_item__ output
    temporal        <bool>      include temporal dimension explicitly instead of buried as channel dimension

    """
    def __init__(self,id,
        num_lags:int=1,
        stimset="Train",
        stims=["Gabor"],
        downsample_s: int=1,
        downsample_t: int=2,
        valid_eye_rad=5.2,
        valid_eye_ctr=(0.0,0.0),
        fixations_only=True,
        dirname='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/',
        download=False,
        cids=None,
        cropidx=None,
        shifter=None,
        preload=False,
        include_eyepos=True,
        dim_order='chwt',
        include_saccades=None, # must be a dict that says how to implement the saccade basis
        include_frametime=None,
        optics=None,
        temporal=True,
        flatten=True,
        device=torch.device('cpu'),
        dtype=torch.float32):
        
        
        self.device = device
        self.dtype = dtype

        if optics is None:
            optics = {'type': 'none', 'sigma': (0,0,0)}

        # check if a specific spike sorting is requested
        chk = [i for i,j in zip(range(len(id)), id) if '_'==j ]

        if len(chk)==0:
            sessname = id
            spike_sorting = None
        else:
            sessname = id[:chk[0]]
            spike_sorting = id[chk[0]+1:]
        
        if flatten: # flatten overrides temporal (which keeps a unit channel dimension)
            temporal = False

        # load data
        self.dirname = dirname
        self.id = id
        self.cropidx = cropidx
        self.fname = get_stim_list(sessname)
        self.sdnorm = 15 # scale stimuli (puts model in better range??)
        self.stimset = stimset
        self.flatten = flatten
        self.dim_order = dim_order

        # check if we need to download the data
        fpath = os.path.join(self.dirname, self.fname)
        if not os.path.exists(fpath):
            print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
            if download:
                print("Downloading set...")
                download_set(sessname, self.dirname)
            else:
                print("Download is False. Exiting...")
                return

        self.fhandle = h5py.File(os.path.join(self.dirname,self.fname), "r")
        self.isopen = True
        self.num_lags = num_lags
        self.downsample_s = downsample_s
        self.downsample_t = downsample_t
        self.fixations_only = fixations_only
        self.include_eyepos = include_eyepos
        self.include_saccades = include_saccades
        self.include_frametime = include_frametime
        self.valid_eye_rad = valid_eye_rad
        self.valid_eye_ctr = valid_eye_ctr
        self.shifter=shifter
        self.temporal = temporal
        self.spike_sorting = spike_sorting
        self.optics = optics

        # sanity check stimuli (all requested stimuli must be keys in the file)
        newstims = []
        for s in range(len(stims)):
            if stims[s] in self.fhandle.keys():
                newstims.append(stims[s])
        print("Found requested stimuli %s" %newstims)
        self.stims = newstims

        # useful info to pull from meta data
        sz = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['size']
        ppd = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['ppd'][0]
        self.centerpix = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['center'][:]
        self.rect = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['rect'][:]
        self.ppd = ppd
        self.NY = int(sz[0]//self.downsample_s)
        self.NX = int(sz[1]//self.downsample_s)
        self.NF = 1 # number of channels
        self.frate = self.fhandle[self.stims[0]]['Test']['Stim'].attrs['frate'][0]

        # get valid indices
        self.valid = [self.get_valid_indices(stim) for stim in self.stims]
        self.lens = [len(v) for v in self.valid]
        indices = [[i] * v for i, v in enumerate(self.lens)]
        self.stim_indices = np.asarray(sum(indices, []))
        self.indices_hack = np.arange(0,len(self.stim_indices)) # stupid conversion from slice/int/range to numpy array

        # pre-load frame times        
        self.frame_time = np.zeros(len(self.stim_indices))
        
        for ii,stim in zip(range(len(self.stims)), self.stims):
            inds = self.stim_indices==ii
            self.frame_time[inds] = self.fhandle[stim][self.stimset]['frameTimesOe'][0,self.valid[ii]]
        """
        LOAD FRAME TIMES AS TENT BASIS
        We want to represent time in the experiment as a smoothly varying parameter so we can fit up-and-down states, excitability, artifacts, etc.
        """
        if self.include_frametime is not None:
            print("Loading frame times on basis")
            assert type(self.include_frametime)==dict, "include_frametime must be a dict with keys: 'num_basis', 'full_experiment'"
            
            assert not self.include_frametime['full_experiment'], "full_experiment = True is not implemented yet"
            self.frame_basiscenters = np.linspace(np.min(self.frame_time), np.max(self.frame_time), self.include_frametime['num_basis'])
            self.frame_binsize = np.mean(np.diff(self.frame_basiscenters))
            xdiff = np.abs(np.expand_dims(self.frame_time, axis=1) - self.frame_basiscenters)
            self.frame_tents = np.maximum(1-xdiff/self.frame_binsize , 0)
        
        """
        INCLUDE SACCADES
        Build design matrix for saccade lags
        
        include_saccades is a list of dicts. Each dict has arguments for how to construct the basis
        {
            'name': name of the variable
            'basis': actual basis (at the time resolution of the stimulus)
        }
        """
        if self.include_saccades is not None:
            assert type(self.include_saccades)==list, "include_saccades must be a list of dicts with keys: 'name', 'offset', 'basis'"
            from scipy.signal import fftconvolve
            print("Loading saccade times on basis")
            self.saccade_times = []
            for isacfeature in range(len(self.include_saccades)):
                self.saccade_times.append( np.zeros( (len(self.stim_indices), self.include_saccades[isacfeature]['basis'].shape[1])))

            for ii,stim in zip(range(len(self.stims)), self.stims):
                print("%d) %s" %(ii,stim))
                labels = self.fhandle[stim][self.stimset]['labels'][0,:]
                sactimes = np.diff((labels==2).astype('float32'))
                for isacfeature in range(len(self.include_saccades)):
                    if self.include_saccades[isacfeature]['name']=="sacon":
                        sacstim = (sactimes==1).astype('float32')
                    elif self.include_saccades[isacfeature]['name']=="sacoff":
                        sacstim = (sactimes==-1).astype('float32')

                    # shift forward or backward to make acasual or delayed lags
                    off = self.include_saccades[isacfeature]['offset']
                    sacstim = np.roll(sacstim, off)
                    # zero out invalid after shift
                    if off < 0:
                        sacstim[off:] = 0
                    elif off > 0:
                        sacstim[:off] = 0
                    
                    # convolve with basis
                    sacfull = fftconvolve(np.expand_dims(sacstim, axis=1), self.include_saccades[isacfeature]['basis'], axes=0)

                    # index into valid times
                    inds = self.stim_indices==ii
                    self.saccade_times[isacfeature][inds,:] = sacfull[self.valid[ii],:]



        # setup cropping
        if cropidx:
            self.cropidx = cropidx
            self.NX = cropidx[0][1] - cropidx[0][0]
            self.NY = cropidx[1][1] - cropidx[1][0]
        else:
            self.cropidx = None

        # spike meta data / specify clusters
        if self.spike_sorting is not None:
            self.cluster_ids = self.fhandle['Neurons'][self.spike_sorting]['cids'][0,:]
            cgs = self.fhandle['Neurons'][self.spike_sorting]['cgs'][0,:]
        else:
            self.cluster_ids = self.fhandle[self.stims[0]]['Test']['Robs'].attrs['cids']
            if 'cgs' in self.fhandle['Neurons'].keys():
                cgs = self.fhandle['Neurons']['cgs'][:][0]
            else:
                cgs = np.ones(len(self.cluster_ids))*2
        
        self.NC = len(self.cluster_ids)
        if cids is not None:
            self.cids = cids
            self.NC = len(cids)
            self.cluster_ids = self.cluster_ids[cids]
        else:
            self.cids = list(range(0,self.NC-1))
        
        # self.single_unit = [int(cgs[c])==2 for c in self.cids]

        if self.spike_sorting is not None:
            from V1FreeViewingCode.Analysis.notebooks.Utils import bin_at_frames
            """
            HANDLE specific spike sorting
            """
            st = self.fhandle['Neurons'][spike_sorting]['times'][0,:]
            clu = self.fhandle['Neurons'][spike_sorting]['cluster'][0,:]
            cids = self.cluster_ids

            Robs = np.zeros((len(self.frame_time), self.NC))
            inds = np.argsort(self.frame_time)
            ft = self.frame_time[inds]
            for cc in range(self.NC):
                cnt = bin_at_frames(st[clu==cids[cc]], ft, maxbsize=1.2/self.frate)
                Robs[inds,cc] = cnt
            self.y = torch.tensor(Robs.astype('float32'), device=self.device, dtype=self.dtype)
        
        if preload: # preload data if it will fit in memory
            self.preload=False
            print("Preload True. Loading ")
            n = len(self)
            if self.flatten:
                self.x = torch.ones((n,self.NF*self.NY*self.NX*self.num_lags), device=self.device, dtype=self.dtype)
            else:
                if self.dim_order=='cxyt':
                    self.x = torch.ones((n,self.NF, self.NY, self.NX, self.num_lags), device=self.device, dtype=self.dtype)
                elif self.dim_order=='txy':
                    self.x = torch.ones((n,self.num_lags, self.NY, self.NX), device=self.device, dtype=self.dtype)

            if self.spike_sorting is None:
                self.y = torch.ones((n,self.NC), device=self.device, dtype=self.dtype)
            self.eyepos = torch.ones( (n,2), device=self.device, dtype=self.dtype)
            chunk_size = 10000
            nsteps = n//chunk_size+1
            for i in range(nsteps):
                print("%d/%d" %(i+1,nsteps))
                inds = np.arange(i*chunk_size, np.minimum(i*chunk_size + chunk_size, n))
                sample = self.__getitem__(inds)
                self.x[inds,:] = sample['stim'] #.detach().clone()
                if self.spike_sorting is None:
                    self.y[inds,:] = sample['robs'] # .detach().clone()
                self.eyepos[inds,0] = sample['eyepos'][:,0]#.detach().clone()
                self.eyepos[inds,1] = sample['eyepos'][:,1]#.detach().clone()
            print("Done")
            self.datafilters = torch.ones(self.y.shape, device=self.device, dtype=self.dtype)
            self.preload = True
        self.preload = preload

    def __getitem__(self, index):
        """
            This is a required Dataset method
        """            
        
        if self.preload:
            if self.temporal:
                if type(index)==int:
                    stim = self.x[index,:].unsqueeze(0)
                else:
                    stim = self.x[index,:].unsqueeze(1) # TODO: check that implicit dimensions are handled correctly
            else:
                stim = self.x[index,:]
            
            out = {'stim': stim, 'robs': self.y[index,:], 'eyepos': self.eyepos[index,:], 'dfs': self.datafilters[index,:]}

            if self.include_frametime is not None:
                out['frametime'] = torch.tensor(self.frame_tents[index,:].astype('float32'))
                out['frame_times'] = torch.Tensor(self.frame_time[index,None].astype('float32'))
            
            if self.include_saccades is not None:
                for ii in range(len(self.saccade_times)):
                    out[self.include_saccades[ii]['name']] = torch.tensor(self.saccade_times[ii][index,:].astype('float32'))

            return out
        else:  
            if self.optics['type']=='gausspsf':
                from scipy.ndimage import gaussian_filter

            if type(index)==int: # special case where a single instance is indexed
                inisint = True
            else:
                inisint = False

            # index into valid stimulus indices (this is part of handling multiple stimulus sets)
            uinds, uinverse = np.unique(self.stim_indices[index], return_inverse=True)
            indices = self.indices_hack[index] # this is now a numpy array

            
            # loop over stimuli included in this index
            for ss in range(len(uinds)):
                istim = uinds[ss] # index into stimulus
                stim_start = np.where(self.stim_indices==istim)[0][0]
                # stim_inds = np.where(uinverse==ss)[0] - stim_start
                stim_inds = indices - stim_start
                ix = uinverse==ss
                if inisint:
                    valid_inds = self.valid[istim][stim_inds]
                    file_inds = valid_inds - range(0,self.num_lags*self.downsample_t)
                else:
                    stim_inds = stim_inds[ix]
                    valid_inds = self.valid[istim][stim_inds]
                    file_inds = np.expand_dims(valid_inds, axis=1) - range(0,self.num_lags*self.downsample_t)
                    
                ufinds, ufinverse = np.unique(file_inds.flatten(), return_inverse=True)
                if self.cropidx and not self.shifter:
                    I = self.fhandle[self.stims[istim]][self.stimset]["Stim"][self.cropidx[1][0]:self.cropidx[1][1],self.cropidx[0][0]:self.cropidx[0][1],ufinds]
                else:
                    I = self.fhandle[self.stims[istim]][self.stimset]["Stim"][:,:,ufinds]

                if self.shifter:
                    eyepos = self.fhandle[self.stims[istim]][self.stimset]["eyeAtFrame"][1:3,ufinds].T
                    eyepos[:,0] -= self.centerpix[0]
                    eyepos[:,1] -= self.centerpix[1]
                    eyepos/= self.ppd
                    I = self.shift_stim(I, eyepos)
                    if self.cropidx:
                        I = I[self.cropidx[1][0]:self.cropidx[1][1],self.cropidx[0][0]:self.cropidx[0][1],:]
                
                if self.optics['type']=='gausspsf':
                    I = gaussian_filter(I, self.optics['sigma'])

                if self.spike_sorting is None:
                    R = self.fhandle[self.stims[istim]][self.stimset]["Robs"][:,valid_inds]
                    R = R.T

                if self.include_eyepos:
                    eyepos = self.fhandle[self.stims[istim]][self.stimset]["eyeAtFrame"][1:3,valid_inds].T
                    if inisint:
                        eyepos[0] -= self.centerpix[0]
                        eyepos[1] -= self.centerpix[1]
                    else:    
                        eyepos[:,0] -= self.centerpix[0]
                        eyepos[:,1] -= self.centerpix[1]
                    eyepos/= self.ppd

                sz = I.shape
                
                if inisint:
                    I = np.expand_dims(I, axis=3)

                # expand time dimension to native H x W x B x T
                I = I[:,:,ufinverse].reshape(sz[0],sz[1],-1, self.num_lags*self.downsample_t)

                # move batch dimension into 0th spot and re-order the remaining dims
                if 'hwt' in self.dim_order:
                    I = I.transpose((2,0,1,3))
                elif 'thw' in self.dim_order:
                    I = I.transpose((2,3,0,1))
                else:
                    raise ValueError('PixelDataset: Unknown dim_order, must be thw or hwt')

                if self.spike_sorting is None:
                    if inisint:
                        NumC = len(R)
                    else:
                        NumC = R.shape[1]

                    if self.NC != NumC:
                        if inisint:
                            R = R[np.asarray(self.cids)]
                        else:
                            R = R[:,np.asarray(self.cids)]

                # concatentate if necessary
                if ss ==0:
                    S = torch.tensor(self.transform_stim(I), device=self.device, dtype=self.dtype)
                    if self.spike_sorting is None:
                        Robs = torch.tensor(R, device=self.device, dtype=self.dtype)

                    if self.include_eyepos:
                        ep = torch.tensor(eyepos, device=self.device, dtype=self.dtype)
                    else:
                        ep = None

                    # if inisint:
                    #     S = S[0,:,:,:] #.unsqueeze(0)
                    #     Robs = Robs[0,:] #.unsqueeze(0)
                    #     ep = ep[0,:] #.unsqueeze(0)

                else:
                    S = torch.cat( (S, torch.tensor(self.transform_stim(I), device=self.device, dtype=self.dtype)), dim=0)
                    if self.spike_sorting is None:
                        Robs = torch.cat( (Robs, torch.tensor(R, device=self.device, dtype=self.dtype)), dim=0)
                    if self.include_eyepos:
                        ep = torch.cat( (ep, torch.tensor(eyepos, device=self.device, dtype=self.dtype)), dim=0)

            if self.temporal:
                if inisint:
                    S = S.unsqueeze(0) # add channel dimension
                else:
                    S = S.unsqueeze(1) # add channel dimension

            if self.spike_sorting is not None:
                Robs = self.y[index,:]

            if self.flatten:
                if inisint:
                    S = torch.flatten(S, start_dim=0)
                else:
                    S = torch.flatten(S, start_dim=1)

            out = {'stim': S, 'robs': Robs, 'eyepos': ep, 'dfs': torch.ones(Robs.shape, device=self.device, dtype=self.dtype)}

            if self.include_frametime is not None:
                out['frametime'] = torch.tensor(self.frame_tents[index,:].astype('float32'))
                out['frame_times'] = torch.Tensor(self.frame_time[index,None].astype('float32'))
            
            if self.include_saccades is not None:
                for ii in range(len(self.saccade_times)):
                    out[self.include_saccades[ii]['name']] = torch.tensor(self.saccade_times[ii][index,:].astype('float32'))

            return out

    def __len__(self):
        return sum(self.lens)

    def get_valid_indices(self, stim):
        # get blocks (start, stop) of valid samples
        blocks = self.fhandle[stim][self.stimset]['blocks'][:,:]
        valid = []
        for bb in range(blocks.shape[1]):
            valid.append(np.arange(blocks[0,bb]+self.num_lags*self.downsample_t,
                blocks[1,bb])) # offset start by num_lags
        
        valid = np.concatenate(valid).astype(int)

        if self.fixations_only:
            fixations = np.where(self.fhandle[stim][self.stimset]['labels'][:]==1)[1]
            valid = np.intersect1d(valid, fixations)
        
        if self.valid_eye_rad:
            xy = self.fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
            xy[:,0] -= self.centerpix[0]
            xy[:,1] = self.centerpix[1] - xy[:,1] # y pixels run down (flip when converting to degrees)
            # convert to degrees
            xy = xy/self.ppd
            # subtract offset
            xy[:,0] -= self.valid_eye_ctr[0]
            xy[:,1] -= self.valid_eye_ctr[1]
            eyeCentered = np.hypot(xy[:,0],xy[:,1]) < self.valid_eye_rad
            valid = np.intersect1d(valid, np.where(eyeCentered)[0])

        return valid

    def transform_stim(self, s):
        # stim comes in N,Lags,Y,X
        s = s.astype('float32')/self.sdnorm

        if self.downsample_t>1 or self.downsample_s>1:
            from scipy.ndimage import gaussian_filter
            
            if 'hwt' in self.dim_order:
                sig = [0, self.downsample_s-1, self.downsample_s-1, self.downsample_t-1] # smoothing before downsample
                s = gaussian_filter(s, sig)
                s = s[:,::self.downsample_s,::self.downsample_s,::self.downsample_t]
            elif 'thw' in self.dim_order:
                sig = [0, self.downsample_t-1, self.downsample_s-1, self.downsample_s-1] # smoothing before downsample
                s = gaussian_filter(s, sig)
                s = s[:,::self.downsample_t,::self.downsample_s,::self.downsample_s]
            else:
                raise ValueError('PixelDataset: Unknown dim_order, must be thw or hwt')

        if 'c' == self.dim_order[0]:
            s = np.expand_dims(s, axis=1)

        if s.shape[0]==1:
            s=s[0,:,:,:] # return single item

        return s


    def shift_stim(self, im, eyepos):
        """
        apply shifter to translate stimulus as a function of the eye position
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = im.shape
        eyepos = torch.tensor(eyepos.astype('float32'))
        im = torch.tensor(im[:,None,:,:].astype('float32'))
        im = im.permute((3,1,0,2))

        shift = self.shifter(eyepos).detach()
        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[0], sz[1])), align_corners=True)

        im2 = F.grid_sample(im, grid, align_corners=True)
        im2 = im2[:,0,:,:].permute((1,2,0)).detach().cpu().numpy()

        return im2
    
    def get_null_adjusted_ll_temporal(self,model,batch_size=5000):
        """
        Get raw and null log-likelihood for all time points
        """
        import torch
        loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
        

        nt = len(self)
        llneuron = np.zeros((nt, self.NC))
        llnull = np.zeros((nt, self.NC))
        robs = np.zeros((nt,self.NC))
        robshat = np.zeros((nt,self.NC))
        eyepos = np.zeros((nt,2))

        nsteps = nt//batch_size + 1

        model.cpu()

        print("Getting log-likelihood for all time bins")
        for istep in tqdm(range(nsteps)):

            if nsteps==1:   
                index = np.arange(0,nt)
            else:
                index = (istep-1)*batch_size + np.arange(0, batch_size)
                index = index[index < nt]
                
            sample = self[index]
            try:
                yhat = model(sample['stim'], shifter=sample['eyepos'], sample=sample)
            except TypeError:
                yhat = model(sample['stim'], shifter=sample['eyepos'])

            llneuron[index,:] = -loss(yhat,sample['robs']).detach().cpu().numpy()
            llnull[index,:] = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy()
            robs[index,:] = sample['robs']
            robshat[index,:] = yhat.detach()
            eyepos[index,:] = sample['eyepos']
        
        return {'llraw': llneuron, 'llnull': llnull, 'robshat': robshat, 'robs': robs, 'eyepos': eyepos}

    def get_ll_by_eyepos(self, model, lldict=None, nbins=20, binsize=.5, bounds=[-5,5],
        batch_size=5000, plot=True, use_stim=None):
        """
        get_ll_by_eyepos gets the null-adjusted loglikelihood for each cell as a function of eye position
        """
        if lldict is None:
            lldict = self.get_null_adjusted_ll_temporal(model, batch_size=batch_size)

        import numpy as np
        import matplotlib.pyplot as plt
        print("Getting log-likelihood as a function of eye position")

        bins = np.linspace(bounds[0],bounds[1],nbins)

        LLspace = np.zeros((nbins,nbins,self.NC))

        if plot:
            sx = np.ceil(np.sqrt(self.NC))
            sy = np.round(np.sqrt(self.NC))
            plt.figure(figsize=(3*sx,3*sy))

        for cc in tqdm(range(self.NC)):
            for ii,xx in zip(range(nbins),bins):
                for jj,yy in zip(range(nbins),bins):
                    ix = np.hypot(lldict['eyepos'][:,0] - xx, lldict['eyepos'][:,1]-yy) < binsize
                    if not use_stim is None:
                        ix = np.logical_and(self.stim_indices==use_stim, ix)
                    LLspace[jj,ii,cc] = np.mean(lldict['llraw'][ix,cc]-lldict['llnull'][ix,cc])

            if plot:
                plt.subplot(sx,sy,cc+1)
                plt.imshow(LLspace[:,:,cc], extent=[bounds[0],bounds[1],bounds[0],bounds[1]])
                plt.title(cc)

        return LLspace
    
    def get_null_adjusted_ll(self, model, sample=None, bits=False, use_shifter=True):
        '''
        get null-adjusted log likelihood
        bits=True will return in units of bits/spike
        '''
        m0 = model.cpu()
        loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
        if sample is None:
            sample = self[:]

        lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
        if use_shifter:
            yhat = m0(sample['stim'], shifter=sample['eyepos'])
        else:
            yhat = m0(sample['stim'], sample=sample)
        llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
        rbar = sample['robs'].sum(axis=0).numpy()
        ll = (llneuron - lnull)/rbar
        if bits:
            ll/=np.log(2)
        return ll