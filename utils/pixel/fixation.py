import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from .utils0 import downsample_time
from .utils import get_stim_list, download_set
import dill
from tqdm import tqdm

class h5py_list(list):
    def __getitem__(self, idx):
        return h5py.File(super().__getitem__(idx), 'r')
    def __setitem__(self, idx, val):
        val = val.filename if type(val) is h5py.File else val
        super().__setitem__(idx, val)
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def shift_stim(stim, shift,size,mode='bilinear'):
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
    dy, dx = torch.meshgrid(torch.arange(size[0])-size[0]/2, torch.arange(size[1])-size[1]/2)
    scalex = (stim.shape[2]/2)
    scaley = (stim.shape[3]/2)
    dx = dx / scalex
    dy = dy / scaley
    dx = dx.unsqueeze(0).unsqueeze(-1)
    dy = dy.unsqueeze(0).unsqueeze(-1)

    sx = shift[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/scalex
    sy = shift[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/scaley

    grid = torch.cat((dx+sx, dy+sy), dim=-1)

    return F.grid_sample(stim, grid, mode=mode, align_corners=False), grid

    
class FixationMultiDataset(Dataset):

    def __init__(self,
        sess_list,
        dirname,
        stimset="Train",
        requested_stims=["Gabor"],
        downsample_s: int=1,
        downsample_t: int=1,
        num_lags: int=1,
        num_lags_pre_sac: int=40,
        saccade_basis = None,
        max_fix_length: int=1000,
        download=True,
        flatten=False,
        crop_inds=None,
        spike_sorting='kilo',
        binarize_spikes=False,
        min_fix_length: int=50,
        max_block_length: int=1000,
        valid_eye_rad=5.2,
        add_noise=0,
        use_blocks=False,
        verbose=True):

        self.dirname = dirname
        self.stimset = stimset
        self.requested_stims = requested_stims
        self.downsample_s = downsample_s
        self.downsample_t = downsample_t
        self.spike_sorting = spike_sorting # only one option for now
        self.valid_eye_rad = valid_eye_rad
        self.min_fix_length = min_fix_length
        self.flatten = flatten
        self.num_lags = num_lags
        self.num_lags_pre_sac = num_lags_pre_sac
        self.normalizing_constant = 50
        self.max_fix_length = max_fix_length
        self.max_block_length = max_block_length
        self.saccade_basis = saccade_basis
        self.shift = None # default shift to None. To provide shifts, set outside this class. Should be a list of shift values equal to size dataset.eyepos in every way
        self.add_noise = add_noise
        self.binarize_spikes = binarize_spikes
        
        if self.saccade_basis is not None:
            if type(self.saccade_basis) is np.array:
                self.saccadeB = self.saccade_basis
            else:
                if type(self.saccade_basis) is not dict or 'max_len' not in self.saccade_basis.keys():
                    self.saccade_basis['max_len'] = 40
                if type(self.saccade_basis) is not dict or 'num' not in self.saccade_basis.keys():
                    self.saccade_basis['num'] = 15
                self.saccadeB = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,self.saccade_basis['max_len'])), axis=1) - np.arange(0,self.saccade_basis['max_len'],self.saccade_basis['max_len']/self.saccade_basis['num']))/self.saccade_basis['max_len']*self.saccade_basis['num'], 0)
        else:
            self.saccadeB = None
            
        # find valid sessions
        stim_list = get_stim_list() # list of valid sessions
        new_sess = []
        for sess in sess_list:
            if sess in stim_list.keys():
                if verbose:
                    print("Found [%s]" %sess)
                new_sess.append(sess)

        self.sess_list = new_sess # is a list of valid sessions
        self.fnames = [get_stim_list(sess) for sess in self.sess_list] # is a list of filenames

        # check if files exist. download if they don't
        for isess,fname in enumerate(self.fnames):
            fpath = os.path.join(dirname, fname)
            if not os.path.exists(fpath):
                print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
                if download:
                    print("Downloading set...")
                    download_set(self.sess_list[isess], dirname)
                else:
                    print("Download is False. Exiting...")

        # open hdf5 files as a list of handles
        self.fhandles = h5py_list([os.path.join(dirname, fname) for fname in self.fnames])

        # build index map
        self.file_index = [] # which file the fixation corresponds to
        self.stim_index = [] # which stimulus the fixation corresponds to
        self.fixation_inds = [] # the actual index into the hdf5 file for this fixation
        self.eyepos = []
        
        # for getting blocks instead of fixations
        self.block_file_index = []
        self.block_stim_index = []
        self.block_inds = []

        self.unit_ids_orig = []
        self.unit_id_map = []
        self.unit_ids = []
        self.num_units = []
        self.NC = 0       
        self.time_start = 0
        self.time_stop = 0
        self.use_blocks = use_blocks

        for f, fhandle in enumerate(self.fhandles): # loop over experimental sessions
            
            # store neuron ids
            # unique_units = np.unique(fhandle['Neurons'][self.spike_sorting]['cluster'][:]).astype(int)
            unique_units = fhandle['Neurons'][self.spike_sorting]['cids'][0,:].astype(int)
            self.unit_ids_orig.append(unique_units)
            
            # map unit ids to index into the new ids
            mc = np.max(unique_units)+1
            unit_map = -1*np.ones(mc, dtype=int)
            unit_map[unique_units] = np.arange(len(unique_units))+self.NC
            self.unit_id_map.append(unit_map)

            # number of units in this session
            self.num_units.append(len(self.unit_ids_orig[-1]))

            # new unit ids
            self.unit_ids.append(np.arange(self.num_units[-1])+self.NC)
            self.NC += self.num_units[-1]

            # loop over stimuli
            for s, stim in enumerate(self.requested_stims): # loop over requested stimuli
                if stim in fhandle.keys(): # if the stimuli exist in this session
                    
                    frate = fhandle[stim]['Train']['Stim'].attrs['frate'][0]
                    sz = fhandle[stim]['Train']['Stim'].attrs['size']
                    self.dims = [1, int(sz[0]), int(sz[1])]

                    # get fixationss
                    labels = fhandle[stim][stimset]['labels'][:] # labeled eye positions
                    labels = labels.flatten()
                    labels[0] = 0 # force ends to be 0, so equal number onsets and offsets
                    labels[-1] = 0
                    fixations = np.diff( (labels ==  1).astype(int)) # 1 is fixation
                    fixstart = np.where(fixations==1)[0]
                    fixstop = np.where(fixations==-1)[0]

                    # offset to include lags before fixation onset
                    fixstart = fixstart-self.num_lags_pre_sac
                    fixstop = fixstop[fixstart>=0]
                    fixstart = fixstart[fixstart>=0]

                    nfix = len(fixstart)
                    if verbose:
                        print("%d fixations" %nfix)

                    # get valid indices
                    # get blocks (start, stop) of valid samples
                    num_blocks = fhandle[stim][stimset]['blocks'].shape[1]
                    blocks = fhandle[stim][stimset]['blocks'][:,:]
                    ctr = 1
                    while ctr < num_blocks:
                        b0 = int(blocks[0,ctr])
                        b1 = int(blocks[1,ctr])-1
                        # print(ctr, b1-b0)
                        if (b1 - b0) > max_block_length:
                            # print('splitting block %d' %ctr)
                            # split block
                            blocks[1,ctr] = b0 + max_block_length
                            # print(blocks[:,ctr])
                            blocks = np.concatenate([blocks[:,:ctr+1], np.array([b0+max_block_length, b1])[:,None], blocks[:,ctr+1:]], axis=1)
                            num_blocks += 1
                        ctr += 1


                    for bb in range(num_blocks):
                        b0 = int(blocks[0,bb])
                        b1 = int(blocks[1,bb])-1
                        if b1 - b0 < self.num_lags:
                            continue

                        block_inds = np.arange(b0,b1)
                        # exy = fhandle[stim][stimset]['eyeAtFrame'][:,b0:b1]
                        # ft = fhandle[stim][stimset]['frameTimesOe'][0,b0:b1]
                        self.block_file_index.append(f)
                        self.block_stim_index.append(s)
                        self.block_inds.append(block_inds)
                        
                    valid_inds = []
                    for bb in range(num_blocks):
                        valid_inds.append(np.arange(blocks[0,bb],
                        blocks[1,bb]))
        
                    valid_inds = np.concatenate(valid_inds).astype(int)

                    for fix_ii in range(nfix): # loop over fixations
                        
                        # get the index into the hdf5 file
                        fix_inds = np.arange(fixstart[fix_ii]+1, fixstop[fix_ii])

                        has_block = np.sum(np.logical_and(fix_inds[0] > blocks[0,:], fix_inds[-1] < blocks[1,:]))
                        if has_block == 0:
                            if verbose:
                                print("fixation not in block. skipping %d" %fix_ii)
                            continue

                        # fix_inds = np.intersect1d(fix_inds, valid_inds)
                        # if len(np.where(np.diff(fhandle[stim][stimset]['frameTimesOe'][0,fix_inds])>0.01)[0]) > 1:
                        #     if verbose:
                        #         print("dropped frames. skipping %d" %fix_ii)
                        #     continue

                        if len(fix_inds) > self.max_fix_length:
                            fix_inds = fix_inds[:self.max_fix_length]

                        # check if the fixation meets our requirements to include
                        # sample eye pos
                        ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
                        centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
                        eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,fix_inds].T
                        eye_tmp[:,0] -= centerpix[0]
                        eye_tmp[:,1] -= centerpix[1]
                        eye_tmp/= ppd

                        if len(fix_inds) < self.min_fix_length:
                            if verbose:
                                print("fixation too short. skipping %d" %fix_ii)
                            continue
                        
                        # is the eye position outside the valid region?
                        if np.mean(np.hypot(eye_tmp[(self.num_lags_pre_sac+5):,0], eye_tmp[(self.num_lags_pre_sac+5):,1])) > self.valid_eye_rad:
                            if verbose:
                                print("eye outside valid region. skipping %d" %fix_ii)
                            continue

                        dx = np.diff(eye_tmp, axis=0)
                        vel = np.hypot(dx[:,0], dx[:,1])
                        vel[:self.num_lags_pre_sac+5] = 0
                        # find missed saccades
                        potential_saccades = np.where(vel[5:]>0.1)[0]
                        if len(potential_saccades)>0:
                            sacc_start = potential_saccades[0]
                            valid = np.arange(0, sacc_start)
                        else:
                            valid = np.arange(0, len(fix_inds))
                        
                        # frame time check 
                        frame_times = fhandle[stim][stimset]['frameTimesOe'][0,fix_inds[valid]]
                        # if np.any(np.diff(frame_times)>(1.1/frate)):
                        #     if verbose:
                        #         print("dropped frames second check. skipping %d" %fix_ii)
                        #     continue

                        if len(valid)>self.min_fix_length:
                            self.eyepos.append(eye_tmp[valid,:])
                            self.fixation_inds.append(fix_inds[valid])
                            self.file_index.append(f) # which datafile does the fixation correspond to
                            self.stim_index.append(s) # which stimulus does the fixation correspond to
            
            from copy import deepcopy
            self.orig_dims = deepcopy(self.dims)
            if crop_inds is None:
                self.crop_inds = [0, self.dims[1], 0, self.dims[2]]
            else:
                self.crop_inds = [crop_inds[0], crop_inds[1], crop_inds[2], crop_inds[3]]
                if crop_inds[1]<0:
                    self.dims[1] = self.orig_dims[1] - crop_inds[0] - np.abs(crop_inds[1])
                else:
                    self.dims[1] = crop_inds[1]-crop_inds[0]
                
                if crop_inds[3]<0:
                    self.dims[2] = self.orig_dims[2] - crop_inds[2] - np.abs(crop_inds[3])
                else:
                    self.dims[2] = crop_inds[3]-crop_inds[2]
    
    def block_inds_to_fix_inds(self, block_inds):
        fix_finder = np.array([fix_i for fix_i, fix in enumerate(self.fixation_inds) for _ in fix], dtype=int)
        flat_fix_inds = np.concatenate(self.fixation_inds)
        inds = [file_ind for block_ind in block_inds for file_ind in self.block_inds[block_ind]]
        out_inds = []
        while inds:
            fix_ind = np.where(flat_fix_inds == inds.pop(0))[0]
            if len(fix_ind):
                fix_ind = [fix_finder[i] for i in fix_ind]
                for i in fix_ind:
                    intersection = np.intersect1d(self.fixation_inds[i], inds)
                    if len(intersection) >= len(self.fixation_inds[i])-50:
                        out_inds.append(i)
                        for ind in intersection:
                            inds.remove(ind)
        return out_inds
        
    def __getitem__(self, index):
        """
        Get item for a Fixation dataset.
        Each element in the index corresponds to a fixation. Each fixation will have a variable length.
        Concatenate fixations along the batch dimension.

        """
        stim = []
        robs = []
        dfs = []
        eyepos = []
        frames = []
        sacB = []
        fix_n = []
        

        # handle indices (can be a range, list, int, or slice). We need to convert ints, and slices into an iterable for looping
        if isinstance(index, int) or isinstance(index, np.int64):
            index = [index]
        elif isinstance(index, slice):
            if self.use_blocks:
                index = np.arange(index.start or 0, index.stop or len(self.block_inds), index.step or 1)
            else:
                index = list(range(index.start or 0, index.stop or len(self.fixation_inds), index.step or 1))
        
        # loop over fixations
        for ii in index:

            if self.use_blocks:
                file = self.block_file_index[ii]
                stimix = self.block_stim_index[ii]
                inds = self.block_inds[ii]
            else:    
                file = self.file_index[ii]
                stimix = self.stim_index[ii]
                inds = self.fixation_inds[ii]

            stim_, robs_, dfs_, frames_, eyepos_, sacc_ = self._get_batch(file, stimix, inds)

            stim.append(stim_)
            robs.append(robs_)
            dfs.append(dfs_)
            sacB.append(sacc_)
            eyepos.append(eyepos_)
            frames.append(frames_)
            # fix_n.append(fix_n_)


        # concatenate along batch dimension
        stim = torch.cat(stim, dim=0)
        # fix_n = torch.cat(fix_n, dim=0)
        eyepos = torch.cat(eyepos, dim=0)
        robs = torch.cat(robs, dim=0)
        if self.binarize_spikes:
            robs = (robs>0).float()
        dfs = torch.cat(dfs, dim=0)
        frames = torch.cat(frames, dim=0)

        if self.flatten:
            stim = torch.flatten(stim, start_dim=1)
        
        sample = {'stim': stim, 'robs': robs, 'dfs': dfs, 'eyepos': eyepos, 'frame_times': frames} #'fix_n': fix_n

        if self.saccadeB is not None:
            sample['saccade'] = torch.cat(sacB, dim=0)

        return sample
    
    def _get_stimulus(self, file, stimix, inds):

        """ STIMULUS """
        # THIS is the only line that matters for sampling the stimulus if your file is already set up right
        I = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'][:,:,inds]
        
        # bring the individual values into a more reasonable range (instead of [-127,127])
        I = I.astype(np.float32)/self.normalizing_constant
        
        I = torch.tensor(I, dtype=torch.float32).permute(2,1,0) # [W,H,N] -> [N,H,W]

        if self.add_noise>0:
            I += torch.randn(I.shape)*self.add_noise
        
        I = I.unsqueeze(1)
        
        I = I[...,self.crop_inds[0]:self.crop_inds[1],self.crop_inds[2]:self.crop_inds[3]]
        return I
    
    def _bin_spikes_at_times(self, file, times):
        """ BIN SPIKES AT TIMES """


        spike_inds = np.where(np.logical_and(
            self.fhandles[file]['Neurons'][self.spike_sorting]['times']>=times[0],
            self.fhandles[file]['Neurons'][self.spike_sorting]['times']<=times[-1]+0.01)
            )[1]

        st = self.fhandles[file]['Neurons'][self.spike_sorting]['times'][0,spike_inds]
        clu = self.fhandles[file]['Neurons'][self.spike_sorting]['cluster'][0,spike_inds].astype(int)
        # only keep spikes that are in the requested cluster ids list
        ix = np.in1d(clu, self.unit_ids_orig[file])
        st = st[ix]
        clu = clu[ix]
        # map cluster id to a unit number
        clu = self.unit_id_map[file][clu]

        robs_tmp = torch.sparse_coo_tensor( np.asarray([np.digitize(st, times)-1, clu]),
                np.ones(len(clu)), (len(times), self.NC) , dtype=torch.float32)
        return robs_tmp.to_dense()

    def _get_robs(self, file, stimix, inds):
        # NOTE: normally this would look just like the line above, but for 'Robs', but I am operating with spike times here
        frame_times = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['frameTimesOe'][0,inds]

        frame_times = np.expand_dims(frame_times, axis=1)
        if self.downsample_t>1:
            frame_times = downsample_time(frame_times, self.downsample_t, flipped=False)

        # frames.append(torch.tensor(frame_times, dtype=torch.float32))
        frames = torch.tensor(frame_times, dtype=torch.float32)
        frame_times = frame_times.flatten()

        robs_tmp = self._bin_spikes_at_times(file, frame_times)
        # spike_inds = np.where(np.logical_and(
        #     self.fhandles[file]['Neurons'][self.spike_sorting]['times']>=frame_times[0],
        #     self.fhandles[file]['Neurons'][self.spike_sorting]['times']<=frame_times[-1]+0.01)
        #     )[1]

        # st = self.fhandles[file]['Neurons'][self.spike_sorting]['times'][0,spike_inds]
        # clu = self.fhandles[file]['Neurons'][self.spike_sorting]['cluster'][0,spike_inds].astype(int)
        # # only keep spikes that are in the requested cluster ids list
        # ix = np.in1d(clu, self.unit_ids_orig[file])
        # st = st[ix]
        # clu = clu[ix]
        # # map cluster id to a unit number
        # clu = self.unit_id_map[file][clu]
        
        # # do the actual binning
        # # robs_tmp = bin_population(st, clu, frame_times, self.unit_ids[file], maxbsize=1.2/240)
        
        # # if self.downsample_t>1:
        # #     robs_tmp = downsample_time(robs_tmp, self.downsample_t, flipped=False)

        # robs_tmp = torch.sparse_coo_tensor( np.asarray([np.digitize(st, frame_times)-1, clu]),
        #         np.ones(len(clu)), (len(frame_times), self.NC) , dtype=torch.float32)
        # robs_tmp = robs_tmp.to_dense()

        """ DATAFILTERS """
        nt = len(frame_times)
        NCbefore = int(np.asarray(self.num_units[:file]).sum())
        NCafter = int(np.asarray(self.num_units[file+1:]).sum())
        dfs_tmp = torch.cat(
            (torch.zeros( (nt, NCbefore), dtype=torch.float32),
            torch.ones( (nt, self.num_units[file]), dtype=torch.float32),
            torch.zeros( (nt, NCafter), dtype=torch.float32)),
            dim=1)
        dfs_tmp[:self.num_lags,:] = 0 # temporal convolution will be invalid for the filter length

        return robs_tmp, dfs_tmp, frames
    
    def _get_eyepos(self, file, stimix, inds):
        """ EYE POSITION """
        ppd = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'].attrs['ppd'][0]
        centerpix = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'].attrs['center'][:]
        eye_tmp = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['eyeAtFrame'][1:3,inds].T
        eye_tmp[:,0] -= centerpix[0]
        eye_tmp[:,1] -= centerpix[1]
        eye_tmp/= ppd

        # assert np.all(self.eyepos[ifix] == eye_tmp), 'eyepos does not match between object and file'
        
        if self.downsample_t>1:
            eye_tmp = downsample_time(eye_tmp, self.downsample_t, flipped=False)

        # eyepos.append(torch.tensor(eye_tmp, dtype=torch.float32))
        eyepos = torch.tensor(eye_tmp, dtype=torch.float32)
        return eyepos

    def _get_fixation_inds(self, ifix):

        fix_inds = self.fixation_inds[ifix] # indices into file for this fixation
        file = self.file_index[ifix]
        stimix = self.stim_index[ifix] # stimulus index for this fixation

        return fix_inds, file, stimix

    def _get_block_inds(self, ii):

        file = self.block_file_index[ii]
        stimix = self.block_stim_index[ii]
        inds = self.block_inds[ii]

        return inds, file, stimix

    def _get_batch(self, file, stimix, fix_inds):
        
        """ STIMULUS """
        I = self._get_stimulus(file, stimix, fix_inds)

        nt = I.shape[0]
        # fix_n = torch.ones(nt, dtype=torch.int64)*ifix

        """ SPIKES """
        robs_tmp, dfs_tmp, frames = self._get_robs(file, stimix, fix_inds)
        # robs.append(robs_tmp)

        # if self.downsample_t>1:
        #     dfs_tmp = downsample_time(dfs_tmp, self.downsample_t, flipped=False)
        """ EYE POSITION """
        eyepos = self._get_eyepos(file, stimix, fix_inds)

        # dfs.append(dfs_tmp)

        """ SACCADES (on basis) """
        if self.saccadeB is not None:
            fix_len = nt
            sacB_len = self.saccadeB.shape[0]
            if fix_len < sacB_len:
                sacc_tmp = torch.tensor(self.saccadeB[:fix_len,:], dtype=torch.float32)
            else:
                sacc_tmp = torch.cat( (torch.tensor(self.saccadeB, dtype=torch.float32),
                    torch.zeros( (fix_len-sacB_len, self.saccadeB.shape[1]), dtype=torch.float32)
                    ), dim=0)
        else:
            sacc_tmp = None
            # sacB.append(sacc_tmp)

        return I, robs_tmp, dfs_tmp, frames, eyepos, sacc_tmp

    def __len__(self):

        if self.use_blocks:
            return len(self.block_inds)
        else:
            return len(self.fixation_inds)

    def get_eyepos_interpolant(self, modifier=lambda x: x):
        '''
        This gets the raw eye trace, smooths it and builds an interpolant function that returns the 
        eye position at any time.

        Inputs:
            modifier - a lambda function that modifies the eye position (e.g., smooths it)
                default: lambda x: x (i.e., no modification)
        Outputs:
            feye - interpolant functions that return the x and y eye position at any time
            e.g., Call feye(timesteps) and it will return the eye position as an N x 2 tensor

        '''
        # extract the eye position for the entire experiment and smooth it and build an interpolant
        from scipy.interpolate import interp1d
        
        eyepos = self.fhandles[0]['ddpi']['eyeposDeg']
        timestamps = self.fhandles[0]['ddpi']['timestamps'][:][0]

        # first-order savgol filter fits a line to
        ex = modifier(eyepos[0,:])
        ey = modifier(eyepos[1,:])
        
        fx = interp1d(timestamps, ex, kind='linear')
        fy = interp1d(timestamps, ey, kind='linear')
        feye = lambda t: torch.cat((torch.tensor(fx(t), dtype=torch.float32), torch.tensor(-fy(t), dtype=torch.float32)), dim=1)
        return feye


    def get_stim_indices(self, stim_name='Gabor'):
        if isinstance(stim_name, str):
            stim_name = [stim_name]
        
        stim_id = [i for i,s in enumerate(self.requested_stims) if s in stim_name]

        if self.use_blocks:
            indices = [i for i,s in enumerate(self.block_stim_index) if s in stim_id]
        else:
            indices = [i for i,s in enumerate(self.stim_index) if s in stim_id]
        
        return indices
    
    def expand_shift(self, fix_shift, fix_inds=None):
        if fix_inds is None:
            assert fix_shift.shape[0]==len(self), 'fix_shift not equal number of fixations. Pass in fix_inds as well.'
            fix_inds = np.arange(len(self)) # index into all fixations
        
        new_shift = []
        for i, fix in enumerate(fix_inds):
            new_shift.append(fix_shift[i].repeat(len(self.fixation_inds[fix]), 1))
        
        new_shift = torch.cat(new_shift, dim=0)
        return new_shift

    def plot_shifter(self, shifter, valid_eye_rad=5.2, ngrid = 100):
        import matplotlib.pyplot as plt
        xx,yy = np.meshgrid(np.linspace(-valid_eye_rad, valid_eye_rad,ngrid),np.linspace(-valid_eye_rad, valid_eye_rad,ngrid))
        xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
        ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

        inputs = torch.cat( (xgrid,ygrid), dim=1)

        xyshift = shifter(inputs).detach().numpy()

        xyshift/=valid_eye_rad/60 # conver to arcmin
        vmin = np.min(xyshift)
        vmax = np.max(xyshift)

        shift = [xyshift[:,0].reshape((ngrid,ngrid))]
        shift.append(xyshift[:,1].reshape((ngrid,ngrid))) 
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(shift[0], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(shift[1], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

        return shift

    def get_shifters(self, plot=False):

        shifters = {}
        for sess in self.sess_list:
            sfname = [f for f in os.listdir(self.dirname) if 'shifter_' + sess in f]
                
            if len(sfname) == 0:
                from pixel.utils import download_shifter
                download_shifter(self.sess_list[0], self.dirname)
            else:
                print("Shifter exists")
                import pickle
                fname = os.path.join(self.dirname, sfname[0])
                shifter_res = pickle.load(open(fname, "rb"))
                shifter = shifter_res['shifters'][np.argmin(shifter_res['vallos'])]

            if plot:
                _ = self.plot_shifter(shifter)
            
            shifters[sess] = shifter

        return shifters
    

    # def shift_stim(self, im, shift, unflatten=False):
    #     """
    #     apply shifter to translate stimulus as a function of the eye position
    #     """
    #     import torch.nn.functional as F
    #     import torch
    #     affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    #     sz = [im.shape[0]] + self.dims

    #     if len(im.shape)==2:
    #         unflatten = True
    #         im = im.reshape(sz)

    #     aff = torch.tensor([[1,0,0],[0,1,0]])

    #     affine_trans = shift[:,:,None]+aff[None,:,:]
    #     affine_trans[:,0,0] = 1
    #     affine_trans[:,0,1] = 0
    #     affine_trans[:,1,0] = 0
    #     affine_trans[:,1,1] = 1

    #     n = im.shape[0]
    #     grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[-2], sz[-1])), align_corners=False)

    #     im2 = F.grid_sample(im, grid, align_corners=False)

    #     if unflatten:
    #         torch.flatten(im2, start_dim=1)

    #     return im2
    
    def save(self, filename):
        dill.dump(self.prepickle(), open(filename, 'wb'))
        
    @staticmethod
    def load(filename):
        return dill.load(open(filename, 'rb')).postpickle()
    
    def prepickle(self):
        self.fhandles = [i.filename for i in self.fhandles]
        return self
    
    def postpickle(self):
        self.fhandles = [h5py.File(i, 'r') for i in self.fhandles]
        return self
    
    def get_stas(self):
        def time_embedding(x, num_lags):
            # x is (time, n)
            # output is (time - num_lags + 1, num_lags, n)
            out = torch.stack([x[i:i+num_lags] for i in range(x.shape[0] - num_lags + 1)], dim=0)
            return out.permute(0,2,1).reshape(len(out), -1)
    
        inds = self.get_stim_indices('Gabor')
        xy = 0
        ny = 0
        for ind in tqdm(inds, smoothing=0):
            data = self[ind]
            if len(data['stim'])<self.num_lags:
                continue
            x = time_embedding(data['stim'].flatten(1), self.num_lags)
            x=x**2
            y = data['robs'][self.num_lags-1:]*data['dfs'][self.num_lags-1:]
            xy += (x.T@y).detach().cpu()
            ny += (y.sum(dim=0)).detach().cpu()

        stas = (xy/ny)
        stas = stas.reshape(self.dims[1:] + [self.num_lags] + [self.NC]).permute(2,0,1,3)

        return stas
    
    def get_data(self, inds):
        # return a dict including the data for every index in inds
        data = {k: [] for k in self[0].keys()}
        for ind in inds:
            for k, v in self[ind].items():
                data[k].append(self[ind][k])
    
    @property
    def nsamples(self):
        if self._nsamples is None:
            self._nsamples = sum([len(i) for i in self])
        return self._nsamples