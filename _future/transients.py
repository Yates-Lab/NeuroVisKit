
def plot_transientsC(model, val_dl, cids, num_lags):
    # val_dl must have batch size 1 and ds must have use_blocks = False
    Nfix = len(val_dl)
    n = []
    nbins = 120
    NC = len(cids)
    start = 0
    rsta = np.nan*np.ones( (Nfix, nbins, NC))
    dfs = np.nan*np.ones( (Nfix, nbins, NC))
    rhat = np.nan*np.ones( (Nfix, nbins, NC))
    esta = np.nan*np.ones( (Nfix, nbins, 2))
    for i,batch in enumerate(val_dl):
        batch = to_device(batch, next(model.parameters()).device)
        n_ = batch['eyepos'].shape[0]
        n.append(n_)
        nt = np.minimum(n_, nbins)
        yhat = model(batch)
        dfs[i,start:nt,:] = batch['dfs'][start:nt,cids].cpu()
        esta[i,start:nt,:] = batch['eyepos'][start:nt,:].cpu()
        rsta[i,start:nt,:] = batch['robs'][start:nt,cids].cpu()
        rhat[i,start:nt,:] = yhat[start:nt,:].detach().cpu().numpy()
        del batch
    
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    f = plt.figure(figsize=(10,10))
    for cc in range(NC):
        plt.subplot(sx,sy,cc+1)
        plt.plot(np.nansum(rsta[:,:,cc]*dfs[:,:,cc], axis=0)/np.nansum(dfs[:,:,cc]), 'k')
        plt.plot(np.nansum(rhat[:,:,cc]*dfs[:,:,cc], axis=0)/np.nansum(dfs[:,:,cc]), 'r')
        plt.xlim([num_lags, nbins])
        plt.axis('off')
        plt.title(cc)
    return rsta, rhat, f

def plot_transients_np(model, val_data, stimid=0, maxsamples=120):
    sacinds = np.where( (val_data['fixation_onset'][:,0] * (val_data['stimid'][:,0]-stimid)**2) > 1e-7)[0]
    nsac = len(sacinds)
    data = val_data

    print("Looping over %d saccades" %nsac)

    NC = len(model.cids)
    sta_true = np.nan*np.zeros((nsac, maxsamples, NC))
    sta_hat = np.nan*np.zeros((nsac, maxsamples, NC))

    for i in tqdm(range(len(sacinds)-1)):
        
        ii = sacinds[i]
        jj = sacinds[i+1]
        n = np.minimum(jj-ii, maxsamples)
        iix = np.arange(ii, ii+n)
        
        sample = {key: data[key][iix,:] for key in ['stim', 'robs', 'dfs', 'eyepos']}

        sta_hat[i,:n,:] = model(sample).detach().numpy()
        sta_true[i,:n,:] = sample['robs'][:,model.cids].detach().numpy()

    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    fig = plt.figure(figsize=(sx*2, sy*2))
    for cc in range(NC):
        
        plt.subplot(sx, sy, cc + 1)
        _ = plt.plot(np.nanmean(sta_true[:,:,cc],axis=0), 'k')
        _ = plt.plot(np.nanmean(sta_hat[:,:,cc],axis=0), 'r')
        plt.axis("off")
        plt.title(cc)

    plt.show()

    return sta_true, sta_hat, fig
