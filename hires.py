import torch
import torch.nn as nn
from models.shifters import Shifter
import numpy as np
import matplotlib.pyplot as plt
import os
from models.utils import plot_stas

def get_max_samples(dataset, device,
    history_size=1,
    nquad=0,
    num_cells=None,
    buffer=1.2):
    """
    get the maximum number of samples that fit in memory

    Inputs:
        dataset: the dataset to get the samples from
        device: the device to put the samples on
    Optional:
        history_size: the history size parameter for LBFGS (scales memory usage)
        nquad: the number of quadratic kernels for a gqm (adds # parameters for every new quad filter)
        num_cells: the number of cells in model (n cells * n parameters)
        buffer: extra memory to keep free (in GB)
    """
    if num_cells is None:
        num_cells = dataset.NC
    
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    free = t - (a+r)

    data = dataset[0]
    # mempersample = data['stim'].element_size() * data['stim'].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    mempersample = 0
    for cov in list(data.keys()):
        mempersample += data[cov].element_size() * data[cov].nelement()

    mempercell = mempersample * (nquad+1) * (history_size + 1)
    buffer_bytes = buffer*1024**3

    maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
    print("# samples that can fit on device: {}".format(maxsamples))
    return maxsamples


def convert_to_fourier(stim, abs=True, batch_size=1000, padding=0, window='flattop'):

    assert padding == 0 or padding == 2, "padding must be 0 or 2"

    input_dims = stim.shape[1:]

    if padding == 2:
        pad_size= [d*2 for d in input_dims[1:]]
        crop_inds = [(int(s/4), int(s-s/4)) for s in pad_size] #
    else:
        pad_size= input_dims[1:] #[d*2 for d in input_dims[1:]]
        crop_inds = [(0, s) for s in pad_size] # [(int(s/4), int(s-s/4)) for s in pad_size] #
    
    if window == 'flattop':
        from scipy.signal import flattop as winfun
    elif window == 'hamming':
        from numpy import hamming as winfun

    window = torch.tensor(np.outer(winfun(input_dims[1]), winfun(input_dims[1])), dtype=torch.float32)

    n = stim.shape[0]
    nbatches = n//batch_size
    from tqdm import tqdm
    for batchi in tqdm(range(nbatches), desc='Converting to Fourier'):
        inds = (batchi)*batch_size + np.arange(batch_size)
        # apply window
        s = torch.einsum('bcwhl,wh->bcwhl', stim[inds,...], window)
        # fft, crop, and shift
        s = torch.fft.fftshift(
            torch.fft.fftn(s, s=pad_size), dim=(2,3,4))[...,crop_inds[0][0]:crop_inds[0][1], crop_inds[1][0]:crop_inds[1][1], crop_inds[2][0]:crop_inds[2][1]]
        if abs:
            s = s.abs()
        stim[inds,...] = s

    return stim

def get_train_val_data(ds, force_data_on_device = True,
    history_size=10,
    seed = 1234,
    ctr_inds=None,
    eye_rad=8,
    maxsamples=None,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    
    if maxsamples == None:
        if device == torch.device('cpu'):
            maxsamples = np.inf
        else:
            maxsamples = get_max_samples(ds, device, history_size=history_size, num_cells=ds.NC, nquad=2)
    train_inds, val_inds = ds.get_train_indices()

    n_val = len(val_inds)
    np.random.seed(seed)
    val_inds = np.random.choice(val_inds, size=n_val, replace=False)

    datalength = len(train_inds) + len(val_inds)
    if datalength < maxsamples:
        print("Yay. Entire dataset can fit on device")
    elif force_data_on_device:
        print("Only {} of {} samples can fit on device".format(maxsamples, datalength))
        train_cutoff = int(np.floor( (maxsamples / datalength) * len(train_inds) ) )
        np.random.seed(seed)
        train_inds = np.random.choice(train_inds, size=train_cutoff, replace=False)

        val_cutoff = int(np.floor( (maxsamples / datalength) * len(val_inds) ) )
        np.random.seed(seed)
        val_inds = np.random.choice(val_inds, size=val_cutoff, replace=False)
        print("train and val inds adjusted to fit on device")
    else:
        print("putting dataset in memory on the cpu")
        dataset_device = torch.device("cpu")
        
    # get train and val data
    if ctr_inds is None:
        ctr_inds = np.where(torch.hypot(ds.covariates['eyepos'][ds.valid_idx,0],ds.covariates['eyepos'][ds.valid_idx,1])<eye_rad)[0]

    train_data = ds[np.intersect1d(train_inds, ctr_inds)]
    train_data['stim'] = torch.flatten(train_data['stim'], start_dim=1)

    n_val = len(val_inds)

    val_data = ds[np.intersect1d(val_inds, ctr_inds)]
    val_data['stim'] = torch.flatten(val_data['stim'], start_dim=1)

    val_inds = np.intersect1d(val_inds, ctr_inds)
    train_inds = np.intersect1d(train_inds, ctr_inds)
    
    print("New stim shape {}".format(train_data['stim'].shape))
    return train_data, val_data, train_inds, val_inds


def fit_glms(ds, cids,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    eye_rad = 8, 
    ctr_inds=None,
    history_size=10,
    d2xs = [0.01, .1, 1, 10],
    locals = [.1, 1, 10, 1000],
    d2ts = [0.01, .1, 1, 10]):
    
    # get training / test sets
    train_data, val_data, _,_ = get_train_val_data(ds, force_data_on_device=True,device=device, history_size=history_size, eye_rad=eye_rad, ctr_inds=ctr_inds)
    
    from models.glm import GLM
    from models.utils import eval_model
    from NDNT.training import LBFGSTrainer

    input_dims = ds.dims + [ds.num_lags]

    NC = len(cids)
    if isinstance(d2xs,list):
        d2xs = np.array(d2xs)
    
    if isinstance(locals,list):
        locals = np.array(locals)
    
    if isinstance(d2ts,list):
        d2ts = np.array(d2ts)

    LLvals = np.zeros( (len(d2xs), len(d2ts), len(locals), NC))
    for i,d2x in enumerate(d2xs):
        for j,d2t in enumerate(d2ts):
            for k,loc in enumerate(locals):
                glm0 = GLM(input_dims=input_dims,
                        cids=cids,
                        reg_vals={'d2x':d2x, 'd2t': d2t, 'glocalx': loc})

                glm0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,glm0.cids].mean(dim=0)*glm0.output_NL.beta) - 1)/glm0.output_NL.beta

                glm0.prepare_regularization()
                glm0.to(device)

                optimizer = torch.optim.LBFGS(glm0.parameters(),
                                history_size=history_size,
                                max_iter=10e3,
                                tolerance_change=1e-9,
                                line_search_fn=None,
                                tolerance_grad=1e-5)

                trainer = LBFGSTrainer(
                    optimizer=optimizer,
                    device=device,
                    max_epochs=1,
                    optimize_graph=True,
                    log_activations=False,
                    set_grad_to_none=False,
                    verbose=0)

                trainer.fit(glm0, train_data)

                llval = eval_model(glm0, val_data)
                LLvals[i,j,k,:] = llval.copy()
                print('d2x: %0.5f, d2t: %0.5f, local: %0.5f | %0.5f' %(d2x, d2t, loc, np.mean(llval)))


    # %% find the best regularization
    inds = np.zeros((3, NC))
    for cc in range(NC):
        inds[:,cc] = np.unravel_index(np.argmax(LLvals[...,cc]), LLvals[...,cc].shape)

    #%% refit with best regularization
    glm0 = GLM(input_dims=input_dims,
                        cids=cids,
                        reg_vals={'d2x': d2xs[inds[0,:].astype(int)], 'd2t': d2ts[inds[1,:].astype(int)], 'glocalx': locals[inds[2,:].astype(int)]})

    glm0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,glm0.cids].mean(dim=0)*glm0.output_NL.beta) - 1)/glm0.output_NL.beta

    glm0.prepare_regularization()
    glm0.to(device)

    optimizer = torch.optim.LBFGS(glm0.parameters(),
                    history_size=10,
                    max_iter=10e3,
                    tolerance_change=1e-9,
                    line_search_fn=None,
                    tolerance_grad=1e-5)

    trainer = LBFGSTrainer(
        optimizer=optimizer,
        device=device,
        optimize_graph=True,
        log_activations=False,
        set_grad_to_none=False,
        verbose=0)

    trainer.fit(glm0, train_data)

    llval = eval_model(glm0, val_data)
    
    ws = glm0.core[0].get_weights()
    ws = np.transpose(ws, (2,0,1,3))
    del(train_data)
    del(val_data)
    torch.cuda.empty_cache()

    return glm0, ws, llval


def fit_energymodel(ds, cids,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    eye_rad = 8, 
    history_size=10,
    ctr_inds=None,
    d2xs = [0.01, .1, 1, 10],
    locals = [.1, 1, 10, 1000],
    d2ts = [0.01, .1, 1, 10]):
    
    # get training / test sets
    train_data, val_data, _,_ = get_train_val_data(ds,
        force_data_on_device=True,
        eye_rad=eye_rad,
        ctr_inds=ctr_inds,
        device=device,
        history_size=history_size)
    
    from models.glm import EnergyModel
    from models.utils import eval_model
    from NDNT.training import LBFGSTrainer

    input_dims = ds.dims + [ds.num_lags]

    NC = len(cids)
    if isinstance(d2xs,list):
        d2xs = np.array(d2xs)
    
    if isinstance(locals,list):
        locals = np.array(locals)
    
    if isinstance(d2ts,list):
        d2ts = np.array(d2ts)

    LLvals = np.zeros( (len(d2xs), len(d2ts), len(locals), NC))
    for i,d2x in enumerate(d2xs):
        for j,d2t in enumerate(d2ts):
            for k,loc in enumerate(locals):
                gqm0 = EnergyModel(input_dims=input_dims,
                    cids=cids,
                    nquad=2,
                    reg_vals={'d2x':d2x, 'd2t': d2t, 'glocalx': loc})

                gqm0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,gqm0.cids].mean(dim=0)*gqm0.output_NL.beta) - 1)/gqm0.output_NL.beta

                gqm0.prepare_regularization()
                gqm0.to(device)

                optimizer = torch.optim.LBFGS(gqm0.parameters(),
                                history_size=history_size,
                                max_iter=10e3,
                                tolerance_change=1e-9,
                                line_search_fn=None,
                                tolerance_grad=1e-5)

                trainer = LBFGSTrainer(
                    optimizer=optimizer,
                    device=device,
                    max_epochs=1,
                    optimize_graph=True,
                    log_activations=False,
                    set_grad_to_none=False,
                    verbose=0)

                trainer.fit(gqm0, train_data)

                llval = eval_model(gqm0, val_data)
                LLvals[i,j,k,:] = llval.copy()
                print('d2x: %0.5f, d2t: %0.5f, local: %0.5f | %0.5f' %(d2x, d2t, loc, np.mean(llval)))


    # find the best regularization
    inds = np.zeros((3, NC))
    for cc in range(NC):
        inds[:,cc] = np.unravel_index(np.argmax(LLvals[...,cc]), LLvals[...,cc].shape)

    # refit with best regularization
    gqm0 = EnergyModel(input_dims=input_dims,
                        cids=cids,
                        nquad=2,
                        reg_vals={'d2x': d2xs[inds[0,:].astype(int)], 'd2t': d2ts[inds[1,:].astype(int)], 'glocalx': locals[inds[2,:].astype(int)]})

    # gqm0.core[0].weight.data = winit[:,gqm0.cids].detach().clone()
    gqm0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,gqm0.cids].mean(dim=0)*gqm0.output_NL.beta) - 1)/gqm0.output_NL.beta

    gqm0.prepare_regularization()
    gqm0.to(device)

    optimizer = torch.optim.LBFGS(gqm0.parameters(),
                    history_size=10,
                    max_iter=10e3,
                    tolerance_change=1e-9,
                    line_search_fn=None,
                    tolerance_grad=1e-5)

    trainer = LBFGSTrainer(
        optimizer=optimizer,
        device=device,
        optimize_graph=True,
        log_activations=False,
        set_grad_to_none=False,
        verbose=0)

    trainer.fit(gqm0, train_data)

    llval = eval_model(gqm0, val_data)
    
    ws = [np.transpose(c.get_weights(), (2,0,1,3)) for c in gqm0.core]
    del(train_data)
    del(val_data)
    torch.cuda.empty_cache()
    
    return gqm0, ws, llval


class StandaloneShifter(nn.Module):

    def __init__(self):
        super().__init__()
        self.shifter = Shifter()
        self.loss = nn.MSELoss()

    def forward(self, x):

        return self.shifter(x)

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        y = batch['shift']

        y_hat = self(batch['stim'])

        loss = self.loss(y_hat, y)

        return {'loss': loss, 'train_loss': loss, 'reg_loss': loss}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['shift']

        y_hat = self(batch['stim'])

        loss = self.loss(y_hat, y)
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}


def get_spatial_power(w, r, sm=0):
    
    spower = w.detach().clone()
    if sm > 1:
        spower = torch.conv2d(spower.unsqueeze(0).permute(3,0,1,2), torch.ones((1,1, sm,sm)), stride=1, padding='same')[0,...].permute(1,2,0)

    # spower = stas[...,cix].std(dim=0)
    spatial_power = torch.einsum('whn,n->wh', spower, r)
    spatial_power[spatial_power < .5*spatial_power.max()] = 0 # remove noise floor
    spatial_power /= spatial_power.sum()

    xx,yy = torch.meshgrid(torch.linspace(-1, 1, spower.shape[0]), torch.linspace(-1, 1, spower.shape[1]))

    ctr_x = (spatial_power * yy).sum().item()
    ctr_y = (spatial_power * xx).sum().item()

    return spatial_power, ctr_x, ctr_y


def get_shifter_from_spatial_power(ds, xax=np.arange(-7, 7, .5), rad = 1.5,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    cids=None, bestlag=None):

    train_data, _, _, _ = get_train_val_data(ds, force_data_on_device=True,
        eye_rad=7.5)

    if cids is None:
        FracDF_include = .2
        cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]

    if bestlag is None:
        sta = (train_data['stim'].T)**2@train_data['robs']
        wfull = sta.reshape(ds.dims + [ds.num_lags] + [ds.NC])
        bestlag = wfull.std(dim=(0,1,2)).max(dim=0)[1]

    from tqdm import tqdm
    n = len(xax)
    xx,yy = np.meshgrid(xax, xax)
    dims = xx.shape
    xx = xx.flatten()
    yy = yy.flatten()


    # mus = np.zeros( (2, ds.NC, len(xx)))
    mus = np.zeros( (2, len(xx)))
    sz = ds.dims[1:]
    xi,yi = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))
    xi = torch.tensor(xi, dtype=torch.float32)
    yi = torch.tensor(yi, dtype=torch.float32)

    rw = ds.covariates['robs'].sum(dim=0)/ds.covariates['robs'].sum()

    spowermap = np.zeros(ds.dims[1:] + [len(xx)])
    ns = np.zeros(len(xx))

    # first get indices of degree with most sample
    for i in tqdm(range(len(xx))):
        # try:
        x0 = xx[i]
        y0 = yy[i]

        dist = np.hypot(train_data['eyepos'][:,0].cpu().numpy() - x0, train_data['eyepos'][:,1].cpu().numpy() - y0)
        
        inds = np.where(dist <= rad)[0]
        ns[i] = len(inds)

    id = np.argmax(ns)
    ns[id]

    x0 = xx[id]
    y0 = yy[id]

    dist = np.hypot(train_data['eyepos'][:,0].cpu().numpy() - x0, train_data['eyepos'][:,1].cpu().numpy() - y0)

    inds = np.where(dist <= rad)[0]

    sta = (train_data['stim'][inds,:].T)**2@train_data['robs'][inds,:]
    wfull = sta.reshape(ds.dims + [ds.num_lags] + [ds.NC])
    w = torch.zeros( ds.dims[1:] + [wfull.shape[-1]] )
    for i in range(len(cids)):
        cc = cids[i]
        wtmp = wfull[0,:,:,int(bestlag[i]), cc].clone()
        wtmp = wtmp - wtmp.min()
        wtmp = wtmp / wtmp.max()

        w[...,cc] = wtmp

    w[w<.8] = 0

    wmask = w.clone().numpy()

    mask = 0

    from scipy.signal import convolve2d
    NC = w.shape[-1]
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.ceil(NC/sx))
    muxy = np.zeros((NC, 2))
    plt.figure(figsize=(20,20))
    for cc in range(NC):
        plt.subplot(sx, sy, cc+1)

        im, cx, cy = get_spatial_power(w[...,cc].unsqueeze(-1), rw[cc].unsqueeze(-1))
        wmask[:,:,cc] = im

        plt.imshow(im)
        plt.axis('off')


    fracnz = np.mean(wmask>0, axis=(0,1))
    plt.plot(fracnz)

    cinds = torch.tensor(np.logical_and(fracnz>0,fracnz<.2))

    plt.figure(figsize=(20,20))
    for cc in np.where(cinds)[0]:
        plt.subplot(sx, sy, cc+1)
        plt.imshow(wmask[:,:,cc])
        plt.axis('off')


    plt.figure(figsize=(20,20))
    for i in tqdm(range(len(xx))):
        # try:
        x0 = xx[i]
        y0 = yy[i]

        dist = np.hypot(train_data['eyepos'][:,0].cpu().numpy() - x0, train_data['eyepos'][:,1].cpu().numpy() - y0)
        
        inds = np.where(dist <= rad)[0]
        ns[i] = len(inds)
        # get STA
        sta = (train_data['stim'][inds,:].T)**2@train_data['robs'][inds,:]
        wfull = sta.reshape(ds.dims + [ds.num_lags] + [ds.NC])
        w = torch.zeros( ds.dims[1:] + [wfull.shape[-1]] )
        for i in range(len(cids)):
            cc = cids[i]
            wtmp = wfull[0,:,:,int(bestlag[i]), cc].clone()
            wtmp = wtmp - wtmp.min()
            wtmp = wtmp / wtmp.max()

            w[...,cc] = wtmp

        w[w<.8] = 0
        
        cix = torch.logical_and(~torch.isnan(w.mean(dim=(0,1))), cinds)
        spower, cx, cy = get_spatial_power(w[...,cix], rw[cix])
        
        spowermap[:,:,i] = spower.detach().cpu().numpy()
        
        mus[0,i] = cx
        mus[1,i] = cy
        plt.subplot(n,n,i+1)
        plt.imshow(spower)
        
    # Build interpolation grid for the peak of the spatial power
    selectivityidx = np.mean(spowermap>0, axis=(0,1))

    muscopy = mus.copy()
    si = np.logical_and(selectivityidx < .4, selectivityidx > 0.05)
    si = np.logical_and(si, ns > 3000)
    muscopy = muscopy * si
    muscopy[0,~si] = np.nan
    muscopy[1,~si] = np.nan

    plt.subplot(1,2,1)
    plt.imshow(muscopy[0,:].reshape(dims))
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(muscopy[1,:].reshape(dims))


    mxy = muscopy - np.nanmean(muscopy,axis=1)[:,None]

    f = plt.plot(mxy[0,:]+xx,mxy[1,:]+yy, '.')
    f = plt.plot(xx,yy, '.')


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(mxy[0,:].reshape(dims))
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(mxy[1,:].reshape(dims))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.sqrt(mxy[0,:]**2 + mxy[1,:]**2).reshape(dims))
    plt.colorbar()

    # build interpolation
    from scipy import interpolate

    sp_x = interpolate.LinearNDInterpolator(np.stack( (xx, yy)).T, mxy[0,:])
    sp_y = interpolate.LinearNDInterpolator(np.stack( (xx, yy)).T, mxy[1,:])


    shift_x = sp_x(ds.covariates['eyepos'].numpy())
    shift_y = sp_y(ds.covariates['eyepos'].numpy())

    invalid = np.where(np.isnan(shift_x + shift_y))[0]
    shift = torch.tensor(np.stack( (shift_x, shift_y)).T, dtype=torch.float32)
    shift[invalid,:] = 0

    # track shifts that have valid outcome
    valid_shifts = np.where(~np.isnan(shift_x[ds.valid_idx] + shift_y[ds.valid_idx]))[0]

    # Now, initialize the shifter from the STA shift points
    from datasets.generic import GenericDataset

    valid = np.where(~np.isnan(shift_x + shift_y))[0]
    shift = torch.tensor(np.stack( (shift_x, shift_y)).T, dtype=torch.float32)

    shiftds = GenericDataset({'stim': ds.covariates['eyepos'][valid,:], 'shift': shift[valid,:]}, device=device)
    # split shiftds into train and valid
    ntest = len(valid)//5
    ntrain = len(valid) - ntest
    train_ds, test_ds = torch.utils.data.random_split(shiftds, (ntrain, ntest))

    from torch.utils.data import DataLoader
    shifttrain_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    shifttest_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

    from NDNT.training import Trainer, EarlyStopping

    smod = StandaloneShifter()
    optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)

    earlystopping = EarlyStopping(patience=5, verbose=False)

    trainer = Trainer(smod, optimizer=optimizer,
        device = device,
        max_epochs=200,
        verbose=1,
        dirpath = os.path.join('./checkpoints/standalone_shifter', 'smod'),
        early_stopping=earlystopping)

    # fit
    trainer.fit(smod, shifttrain_dl, shifttest_dl)

    shift = smod(train_data['eyepos'])
    from datasets.pixel.utils import shift_im
    stim = train_data['stim'].cpu().reshape(ds.dims + [ds.num_lags]).permute(0, 3, 1, 2)
    stimshift = shift_im(stim, shift)

    sta1 = torch.einsum('nlxy,nc->lxyc', stimshift, train_data['robs'].cpu()-train_data['robs'].cpu().mean(dim=0))
    _ = plot_stas(sta1.numpy())

    return smod


def compare_ln_en(datadir, sessname):

    import pickle
    from datasets.pixel import Pixel
    from NDNT.utils import ensure_dir
    from models.utils import eval_model

    outdir = os.path.join(datadir, 'LNENmodels')
    ensure_dir(outdir)
    flist = os.listdir(outdir)
    flist = [i for i in flist if sessname in i]

    if len(flist) > 0:
        fname = os.path.join(outdir, flist[0])
        if os.path.exists(fname):
            print("Loading %s" %sessname)
            with open(fname, 'rb') as f:
                ln_vs_en = pickle.load(f)

    else:
        print("Fitting models for %s" %sessname)
        valid_eye_rad = 9.2
        try:
            ds = Pixel(datadir,
                sess_list=[sessname],
                requested_stims=['Gabor'],
                num_lags=12,
                downsample_t=2,
                download=True,
                valid_eye_rad=valid_eye_rad,
                ctr=np.array([0,0]),
                fixations_only=True,
                load_shifters=True,
                spike_sorting='kilowf',
                )
            spike_sorting = 'kilowf'
        except:
            ds = Pixel(datadir,
                sess_list=[sessname],
                requested_stims=['Gabor'],
                num_lags=12,
                downsample_t=2,
                download=True,
                valid_eye_rad=valid_eye_rad,
                ctr=np.array([0,0]),
                fixations_only=True,
                load_shifters=True,
                spike_sorting='kilo',
                )
            spike_sorting = 'kilo'
            
        print("calculating datafilters")
        ds.compute_datafilters(to_plot=True)
        
        from copy import deepcopy
        dims_orig = deepcopy(ds.dims)

        # get cell inclusion
        FracDF_include = .2
        cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
        # ppd = ds.stim_indices[ds.sess_list[0]][ds.requested_stims[0]]['ppd']
        # get STAs at full stimulus resolution
        '''
        We need to get the full stimulus resolution STAs for each cell under fixation and free-viewing
        '''

        # get STA on the free-viewing Gaborium stimulus
        gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()
        # stas_full = ds.get_stas(inds=gab_inds, square=False)

        # mu, bestlag = plot_stas(stas_full.detach().numpy())

        win_size = 35 # size of crop window in pixels


        sta2 = ds.get_stas(inds=gab_inds, square=True)
        spower = sta2[...,cids].std(dim=0)
        spatial_power = torch.einsum('whn,n->wh', spower, ds.covariates['robs'][:,cids].sum(dim=0)/ds.covariates['robs'][:,cids].sum())
        spatial_power[spatial_power < .5*spatial_power.max()] = 0 # remove noise floor
        spatial_power /= spatial_power.sum()

        xx,yy = torch.meshgrid(torch.arange(0, sta2.shape[1]), torch.arange(0, sta2.shape[2]))

        ctr_x = (spatial_power * yy).sum().item()
        ctr_y = (spatial_power * xx).sum().item()

        plt.imshow(spatial_power.detach().numpy())
        plt.plot(ctr_x, ctr_y, 'rx')

        x0 = int(ctr_x) - int(np.ceil(win_size/2))
        x1 = int(ctr_x) + int(np.floor(win_size/2))
        y0 = int(ctr_y) - int(np.ceil(win_size/2))
        y1 = int(ctr_y) + int(np.floor(win_size/2))

        from copy import deepcopy
        if x0 < 0:
            xoff = deepcopy(-x0)
            x0 += xoff
            x1 += xoff
        
        if y0 < 0:
            yoff = deepcopy(-y0)
            y0 += yoff
            y1 += yoff

        plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], 'r')
        
        ds.crop_idx = [y0,y1,x0,x1]

        fname = os.path.join(outdir, 'LN_vs_EN_%s_%d_%d_%d_%d.pkl' %(sessname, y0,y1,x0,x1))
        # from hires import fit_energymodel, fit_glms
        glm0, ws0, llval0 = fit_glms(ds, cids, d2ts=[.1, 1, 10], d2xs=[.1, 1, 10], locals=[1, 10]) #, ctr_inds=gab_inds
        gqm0, wsEn, llvalEn = fit_energymodel(ds, cids, d2ts=[.1, 1, 10], d2xs=[.1, 1, 10], locals=[1, 10])


        # evaluate models on test data
        dstest = Pixel(datadir,
                sess_list=[sessname],
                stimset='Test',
                requested_stims=['Gabor'],
                num_lags=12,
                downsample_t=2,
                download=True,
                valid_eye_rad=valid_eye_rad,
                ctr=np.array([0,0]),
                fixations_only=True,
                load_shifters=True,
                spike_sorting=spike_sorting,
                )
        dstest.crop_idx = ds.crop_idx
        dstest.compute_datafilters(to_plot=True)
        test_data = dstest[:]
        test_data['stim'] = torch.flatten(test_data['stim'], start_dim=1)

        llval0test = eval_model(glm0.to('cpu'), test_data)
        llvalEntest = eval_model(gqm0.to('cpu'), test_data)

        ln_vs_en = {'glm0': glm0, 'gqm0': gqm0, 'ws0': ws0, 'wsEn': wsEn, 'llval0': llval0, 'llvalEn': llvalEn, 
            'llval0test': llval0test, 'llvalEntest': llvalEntest}

        with open(fname, 'wb') as f:
            pickle.dump(ln_vs_en, f)
        
        torch.cuda.empty_cache()
    print("Done")
    return ln_vs_en

from skimage import measure

def get_mask_from_contour(Im, contour):
    import scipy.ndimage as ndimage    
    # Create an empty image to store the masked array
    r_mask = np.zeros_like(Im, dtype='bool')
    ci = np.round(contour[:, 0]).astype('int')
    cj = np.round(contour[:, 1]).astype('int')
    ci = np.minimum(ci, Im.shape[0]-1)
    cj = np.minimum(cj, Im.shape[1]-1)
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    r_mask[ci, cj] = 1
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


def get_rf_contour(rf, thresh = .5, pad=1):
    
    if pad > 0:
        rf = np.pad(rf, pad, 'constant')

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
    
    # ctr = ctr - pad
    # con = con - pad

    return con, ar, ctr, thresh_

# smooth
from scipy.ndimage.filters import gaussian_filter

# sta summary function
def sta_summary(sta, bestlag, label=None, pad = 5,
    sfs = np.arange(30)*.5, oris = np.arange(22)/22*np.pi,
    plot=True, ppd=37.5,
    contour=None):
    from scipy.interpolate import RectBivariateSpline
    from scipy.fftpack import fft2, fftshift
    
    Im = sta[bestlag,...]
    # calculate spatial power by taking the standard deviation over time
    spower = np.std(sta, axis=0)

    # calculate the spatial "power" by squaring and smoothing the RF
    # Im = stas[int(bestlag[cc]), :,:,cc].detach().numpy()
    # spower = stas[...,cc].std(dim=0).detach().numpy()
    # sda = deepcopy(Im)
    # sda = (sda - np.mean(sda)) / np.std(sda)
    # sda = np.abs(sda)

    sda = spower
    sda = (sda - np.min(sda)) / (np.max(sda) - np.min(sda))
    sda = gaussian_filter(sda, 3)
    sda /= np.max(sda)

    if contour is None:
        con, ar, ctr, thresh = get_rf_contour(sda, thresh=.5)
    else:
        con = contour
        thresh = np.nan
        ar = np.nan
        ctr = np.nan*np.zeros(2)

    r_mask = get_mask_from_contour(Im, con)

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(Im, interpolation='nearest', cmap=plt.cm.gray, extent=[0,Im.shape[0]/ppd,0,Im.shape[1]/ppd])
        if label is not None:
            plt.title(label)
            # plt.imshow(Im*r_mask, interpolation='nearest', cmap=plt.cm.gray, extent=[0,Im.shape[0]/ppd,0,Im.shape[1]/ppd])

        plt.subplot(1,3,2)
        plt.imshow(sda)
        plt.plot(con[:, 1], con[:, 0], 'r')
        plt.plot(ctr[1], ctr[0], 'b*')
        plt.title("thresh = %.2f" % thresh)


    # get the fourier transform of Im
    F = fftshift(np.abs(fft2(Im*r_mask)))
    # get axes for fft
    fx = np.arange(-Im.shape[1]//2, Im.shape[1]//2)*ppd/Im.shape[1]
    fy = np.arange(-Im.shape[0]//2, Im.shape[0]//2)*ppd/Im.shape[0]
    X, Y = np.meshgrid(fx, fy)
    i,j = np.where(F==np.max(F))
    
    if plot:
        plt.subplot(1,3,3)
        plt.imshow(F, origin='lower', interpolation='nearest', cmap=plt.cm.gray,extent=(fx[0], fx[-1], fy[0], fy[-1]))
        plt.plot(fx[j], -fy[i], 'r*')
        plt.title("Freq. Domain")

    # find tuning by interpolating around the peak of the fourier transform
    Fmask = np.zeros_like(F)
    padx, pady = np.meshgrid(np.arange(-pad,pad), np.arange(-pad,pad))
    ipad = i[-1]+padx
    jpad = j[-1]+pady
    ipad[ipad<0] = 0
    ipad[ipad>=F.shape[0]] = F.shape[0]-1
    jpad[jpad<0] = 0
    jpad[jpad>=F.shape[1]] = F.shape[1]-1
    Fmask[ipad,jpad] = 1

    Fmasked = Fmask*F
    Fmasked /= np.sum(Fmasked)
    xhat = np.sum(Fmasked*X)
    yhat = np.sum(Fmasked*Y)

    # normalize to unit vector
    n = np.linalg.norm((xhat,yhat))
    x = xhat/n
    y = yhat/n

    m = y/x

    if plot:
        # tuning slope line
        plt.plot(fx, m*fx, 'r')

        # sfs = np.append(0, .5*1.5**np.arange(10))
        plt.plot(x*sfs, y*sfs, 'b.')
        plt.plot(np.cos(oris)*n, np.sin(oris)*n, 'r.')

    I = RectBivariateSpline(fx, fy, F.T)
    sft = np.asarray([I(x*s, y*s) for s in sfs]).flatten()
    orit = np.asarray([I(np.cos(ori)*n, np.sin(ori)*n) for ori in oris]).flatten()

    summary = {'fxpref': xhat, 'fypref':yhat, 'Im': Im,
        'rmask': r_mask,
        'fx':fx, 'fy':fy, 'oris':oris, 'sfs': sfs,
        'sftuning':sft, 'orituning': orit,
        'center': ctr, 'area': ar,
        'contour': con, 'thresh': thresh
        }

    return summary


def get_stas_sig(data, thresh=0.001):

    robs = data['robs']*data['dfs']

    ny = robs.sum(dim=0)

    stas = torch.einsum('ncxyt,nm->xytm', data['stim'], robs)
    stas /= ny[None,None,None,:]
    stas = stas.permute(2,0,1,3).detach().cpu().numpy()

    num_lags = data['stim'].shape[-1]
    
    # get null sta by offsetting spikes acausally by num_lags
    stas_null = torch.einsum('ncxyt,nm->xytm', data['stim'][num_lags*2:,...], robs[:-num_lags*2,:])
    stas_null /= robs[:-num_lags*2,:].sum(dim=0)[None,None,None,:]
    stas_null = stas_null.permute(2,0,1,3).detach().cpu().numpy()


    mu = np.mean(stas_null, axis=0)
    thigh = np.quantile( (stas_null-mu[None,:,:,:]), 1-(thresh/2/num_lags), axis=(0,1,2))
    tlow = np.quantile( (stas_null-mu[None,:,:,:]), thresh/2/num_lags, axis=(0,1,2))

    sig = np.mean((stas - mu[None,:,:,:])>thigh[None,None,None,:], axis=(0,1,2))
    sig = sig + np.mean((stas - mu[None,:,:,:])<tlow[None,None,None,:], axis=(0,1,2))
    plt.plot(sig, '-o')
    plt.axhline(thresh*2)

    s = stas - mu[None,:,:,:]
    
    _, bestlag = plot_stas(s, plot=False)

    num_sig = np.sum(sig > thresh*2)
    print('%d/%d significant RFs' %(num_sig, len(sig)))

    return {'stas': s, 'sig': sig, 'thresh': thresh, 'nspikes': ny, 'bestlag':bestlag}

def get_stas_sig_square(data, thresh=0.001):

    robs = data['robs']*data['dfs']

    ny = robs.sum(dim=0)

    stas = torch.einsum('ncxyt,nm->xytm', data['stim']**2, robs)
    stas /= ny[None,None,None,:]
    stas = stas.permute(2,0,1,3).detach().cpu().numpy()

    num_lags = data['stim'].shape[-1]
    
    # get null sta by offsetting spikes acausally by num_lags
    stas_null = torch.einsum('ncxyt,nm->xytm', data['stim'][num_lags*2:,...]**2, robs[:-num_lags*2,:])
    stas_null /= robs[:-num_lags*2,:].sum(dim=0)[None,None,None,:]
    stas_null = stas_null.permute(2,0,1,3).detach().cpu().numpy()


    mu = np.mean(stas_null, axis=0)
    thigh = np.quantile( (stas_null-mu[None,:,:,:]), 1-(thresh/2/num_lags), axis=(0,1,2))
    tlow = np.quantile( (stas_null-mu[None,:,:,:]), thresh/2/num_lags, axis=(0,1,2))

    sig = np.mean((stas - mu[None,:,:,:])>thigh[None,None,None,:], axis=(0,1,2))
    sig = sig + np.mean((stas - mu[None,:,:,:])<tlow[None,None,None,:], axis=(0,1,2))
    plt.plot(sig, '-o')
    plt.axhline(thresh*2)

    s = stas - mu[None,:,:,:]
    
    _, bestlag = plot_stas(s, plot=False)

    num_sig = np.sum(sig > thresh*2)
    print('%d/%d significant RFs' %(num_sig, len(sig)))

    return {'stas': s, 'sig': sig, 'thresh': thresh, 'nspikes': ny, 'bestlag':bestlag}

def shift_stim(im, shift):
    """
    apply shifter to translate stimulus as a function of the eye position
    """
    import torch.nn.functional as F
    import torch
    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    sz = im.shape # [N x C x W x H x T]

    im = im.permute(0,1,4,2,3).reshape(sz[0], sz[1]*sz[4], sz[2], sz[3])

    sz2 = im.shape
    print(sz2)

    aff = torch.tensor([[1,0,0],[0,1,0]])

    affine_trans = shift[:,:,None]+aff[None,:,:]
    affine_trans[:,0,0] = 1
    affine_trans[:,0,1] = 0
    affine_trans[:,1,0] = 0
    affine_trans[:,1,1] = 1

    n = sz2[0]
    c = sz2[1]

    grid = F.affine_grid(affine_trans, torch.Size((n, c, sz2[-2], sz2[-1])), align_corners=False)

    im2 = F.grid_sample(im, grid, align_corners=False)
    im2 = im2.reshape(sz[0], sz[1], sz[4], sz[2], sz[3]).permute(0, 1, 3, 4, 2)

    return im2
    
def fig04_rf_analysis(sessname = '20220610',
    datadir='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/', overwrite=False):
    
    import pickle
    from NDNT.utils import ensure_dir
    from datasets.pixel import Pixel

    outdir = os.path.join(datadir, 'fig04_hires')
    ensure_dir(outdir)
    flist = os.listdir(outdir)
    # flist = [i for i in flist if sessname in i]
    
    fname = 'hires_' + sessname + '.pkl'
    fnamefull = os.path.join(outdir, fname)
    if fname in flist and not overwrite:
        
        if os.path.exists(fnamefull):
            print("Loading %s" %sessname)
            with open(fnamefull, 'rb') as f:
                rfsum = pickle.load(f)
                return rfsum

    valid_eye_rad = 9.2
    spike_sorting = 'kilowf'
    try:
        ds = Pixel(datadir,
            sess_list=[sessname],
            requested_stims=['Dots', 'Gabor'],
            num_lags=12,
            downsample_t=2,
            download=True,
            valid_eye_rad=valid_eye_rad,
            ctr=np.array([0,0]),
            fixations_only=True,
            load_shifters=True,
            spike_sorting=spike_sorting,
            )
    except:
        spike_sorting = 'kilo'
        ds = Pixel(datadir,
            sess_list=[sessname],
            requested_stims=['Dots', 'Gabor'],
            num_lags=12,
            downsample_t=2,
            download=True,
            valid_eye_rad=valid_eye_rad,
            ctr=np.array([0,0]),
            fixations_only=True,
            load_shifters=True,
            spike_sorting=spike_sorting,
            )
        
    print("Done")
    from copy import deepcopy

    # Find position on screen that has the most samples (resting gaze of the subject)
    rad = 2 # window around each position to count samples
    xax = np.arange(-(valid_eye_rad-1), (valid_eye_rad-1), .5) # grid up space

    xx,yy = np.meshgrid(xax, xax)
    dims = xx.shape
    xx = xx.flatten()
    yy = yy.flatten()
    n = len(xx)

    ns = np.zeros(n)
    for i in range(n):
        # try:
        x0 = xx[i]
        y0 = yy[i]

        dist = np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - x0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - y0)
        
        inds = np.where(dist <= rad)[0]
        ns[i] = len(inds)

    id = np.argmax(ns)
    # resting gaze cetner
    rgx0 = xx[id]
    rgy0 = yy[id]

    print("Found %d samples at %.3f, %.3f" %(ns[id], rgx0, rgy0))

    plt.figure()
    plt.imshow(ns.reshape(dims), cmap='coolwarm', extent=[-valid_eye_rad, valid_eye_rad, valid_eye_rad, -valid_eye_rad])
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel('horizontal position (deg)')
    plt.ylabel('vertical position (deg)')
    plt.plot(rgx0, rgy0, 'o', color='k')

    rad = 6
    ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
    print("computing STAs with %d indices" %len(ctr_inds))

    data = ds[ctr_inds]

    rf = get_stas_sig(data)
    rf2  = get_stas_sig_square(data)

    # summarize linear RFs
    sig = rf['sig']
    # cids = np.where(sig>thresh)[0]
    NCtot = len(sig)
    num_samples = len(ds)
    frate = ds.stim_indices[ds.sess_list[0]]['Gabor']['frate']
    rect = ds.stim_indices[ds.sess_list[0]]['Gabor']['rect']
    ppd = ds.stim_indices[ds.sess_list[0]]['Gabor']['ppd']
    extent = [rect[0]/ppd, rect[2]/ppd, rect[3]/ppd, rect[1]/ppd]
    
    shifter = ds.get_shifters()

    del data
    del ds
    ds = Pixel(datadir,
            sess_list=[sessname],
            requested_stims=['Dots', 'Gabor'],
            num_lags=12,
            downsample_t=2,
            download=True,
            valid_eye_rad=valid_eye_rad,
            ctr=np.array([0,0]),
            fixations_only=True,
            load_shifters=False,
            spike_sorting=spike_sorting,
            )
    ds.shift = None
    # rfnoshift = get_stas_sig(data)
    rfnoshift = ds.get_stas(inds=ctr_inds)
    
    rfsum = {'sessname': sessname,
        'rflin': rf,
        'rfsquare': rf2,
        'rflinns': rfnoshift,
        'shifter': shifter[sessname],
        'NC': NCtot,
        'num_samples': num_samples,
        'frate': frate,
        'rect': rect,
        'extent': extent,
        'ppd': ppd}

    with open(fnamefull, 'wb') as f:
        pickle.dump(rfsum, f)

    return rfsum

    # sum0 = []
    # for i, cc in enumerate(cids):
    #     sum0.append(sta_summary(rf['stas'][...,cc], int(rf['bestlag'][cc]), label="Free Viewing", plot=False))

    # thresh = np.asarray([s['thresh'] for s in sum0])
    # ctrx = np.asarray([s['center'][1] for s in sum0])
    # ctry = np.asarray([s['center'][0] for s in sum0])
    # ar = np.asarray([s['area'] for s in sum0])

    # cx = (ctrx + rect[0])/ppd
    # cy = (ctry + rect[0])/ppd

    # ix = np.where(thresh < .8)[0]

    # NC = len(ix)
    # sx = int(np.ceil(np.sqrt(NC)))
    # sy = int(np.round(np.sqrt(NC)))

    # extent = [rect[0]/ppd, rect[2]/ppd, rect[3]/ppd, rect[1]/ppd]

    # plt.figure(figsize=(sx, sy))
    # for i,cc in enumerate(ix):
    #     plt.subplot(sx, sy, i + 1)
    #     plt.imshow(sum0[cc]['Im'].T, interpolation='none', extent=extent, cmap=plt.cm.gray)
        
    #     # plt.plot(cx[cc], cy[cc], '.r')
    #     plt.plot( sum0[cc]['contour'][:,0]/ppd+extent[0], sum0[cc]['contour'][:,1]/ppd+extent[3], 'r', linewidth=.25)
    #     # plt.plot( (sum0[cc]['contour'][:,1]+rect[1])/ppd, (sum0[cc]['contour'][:,0]+rect[0])/ppd, 'r', linewidth=.2)
    #     plt.axis("tight")

    


