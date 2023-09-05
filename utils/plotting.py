import numpy as np
import imageio, os
import matplotlib.pyplot as plt

def plot_stas(stas, show_zero=True, plot=True, thresh=None, title=None):
    
    NC = stas.shape[-1]
    num_lags= stas.shape[0]

    sx = int(np.ceil(np.sqrt(NC*2)))
    sy = int(np.round(np.sqrt(NC*2)))
    mod2 = sy % 2
    sy += mod2
    sx -= mod2
    mu = np.zeros((NC,2))
    amp = np.zeros(NC)
    blag = np.zeros(NC)

    if plot:
        fig = plt.figure(figsize=(sx*3,sy*2))
    else:
        fig = None

    for cc in range(NC):
        w = stas[:,:,:,cc]

        wt = np.std(w, axis=0)
        wt /= np.max(np.abs(wt)) # normalize for numerical stability
        # softmax
        wt = wt**10
        wt /= np.sum(wt)
        sz = wt.shape
        xx,yy = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))

        mu[cc,0] = np.minimum(np.maximum(np.sum(xx*wt), -.5), .5) # center of mass after softmax
        mu[cc,1] = np.minimum(np.maximum(np.sum(yy*wt), -.5), .5) # center of mass after softmax

        w = (w -np.mean(w) )/ np.std(w)

        bestlag = np.argmax(np.std(w.reshape( (num_lags, -1)), axis=1))
        blag[cc] = bestlag
        
        v = np.max(np.abs(w))
        amp[cc] = np.std(w[bestlag,:,:].flatten())

        if plot:
            plt.subplot(sx,sy, cc*2 + 1)
            plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm_r", extent=(-1,1,-1,1))
            plt.title(cc)
        
        if plot:
            try:
                plt.subplot(sx,sy, cc*2 + 2)
                i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
                t1 = stas[:,i[0],j[0],cc]
                plt.plot(t1, '-ob')
                i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
                t2 = stas[:,i[0],j[0],cc]
                plt.plot(t2, '-or')
                if show_zero:
                    plt.axhline(0, color='k')
                    if thresh is not None:
                        plt.axhline(thresh[cc],color='k', ls='--')
            except:
                pass
        
        if plot and title is not None:
            plt.suptitle(title)
    
    return mu, blag.astype(int), fig

def plot_sta_movie(stas, path='sta.gif', threeD=False, frameDelay=0, is_weights=True, cmap="jet"):
    if is_weights:
        # stas shape is (height, width, num_lags, cids)
        stas = stas.transpose(2, 0, 1, 3)
    NC = stas.shape[-1]
    # stas shape is (num_lags, height, width, cids)
    if frameDelay:
        from scipy.interpolate import interp1d
        num_frames = stas.shape[0]
        num_frames += (num_frames - 1) * frameDelay
        stas_copy = np.empty((num_frames, *stas.shape[1:]))
        for cc in range(NC):
            values = stas[:,:,:,cc]
            xi = np.arange(values.shape[0])
            xi0 = np.linspace(0, stas.shape[0]-1, num_frames)
            interpolator = interp1d(xi, values, axis=0)
            stas_copy[..., cc] = interpolator(xi0)
        stas = stas_copy

    with plt.ioff():
        
        num_lags = stas.shape[0]

        sx = int(np.ceil(np.sqrt(NC)))
        sy = int(np.ceil(NC/sx))
        v_mins = stas.min(axis=(0,1,2))
        v_maxs = stas.max(axis=(0,1,2))

        images = []
        for i in range(num_lags):
            fig = plt.figure(figsize=(sx,sy))
            for cc in range(NC):
                w = stas[i,:,:,cc]
                if threeD:
                    ax = plt.subplot(sx,sy, cc + 1, projection='3d')
                    X, Y = np.indices(w.shape)
                    ax.contour3D(X, Y, w, 50, cmap=cmap, vmin=v_mins[cc], vmax=v_maxs[cc])
                    ax.set_zlim(v_mins[cc], v_maxs[cc])
                else:
                    plt.subplot(sx,sy, cc + 1)
                    plt.imshow(w, interpolation=None, vmin=v_mins[cc], vmax=v_maxs[cc], cmap=cmap, extent=(-1,1,-1,1))
            plt.tight_layout()
            fig.savefig('temp_trash.png')
            plt.close(fig)
            images.append(imageio.imread('temp_trash.png'))
        imageio.mimsave(path, images)
        os.remove('temp_trash.png')
    
