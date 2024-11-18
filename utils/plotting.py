#this file should not have dependencies within this folder.
import numpy as np
import imageio, os
import matplotlib.pyplot as plt
from IPython.display import Video
from tqdm import tqdm

from NeuroVisKit._utils.postprocess import get_conv_submodules

def plot_model_conv(model):
    """Plots the convolutional kernels of a model.
    
    Recommended only for small models.
    """
    f = []
    submods = get_conv_submodules(model)
    for ind, module in enumerate(submods): #model.modules()
        # check if convolutional layer
        # if issubclass(type(module), nn.modules.conv._ConvNd):
        w = module.weight.detach().data.cpu().numpy()
        if len(w.shape) == 5: # 3d conv (cout, cin, x, y, t)
            w = w.mean(1) # asume cin is 1
            w = w/(np.abs(w).max((1, 2, 3), keepdims=True)+1e-8) # normalize all |xyt (1, 2, 3)
        elif len(w.shape) == 4: # 2d conv (cout, cin, x, y)
            w = w/(np.abs(w).max((1, 2, 3), keepdims=True)+1e-8) # normalize all |cin xy (1, 2, 3)
        # shape is (cout, cin, x, y)
        titles = ['cout %d'%i for i in range(w.shape[0])]
        ft = plot_grid(w, titles=titles, suptitle='Layer %d'%ind, desc='Layer %d'%ind, vmin=-1, vmax=1)
        f.append(ft)
    return f

def plot_grid1d(mat, titles=None, desc='Grid plot', plotter=plt.plot, width=5, height=1, ind=False, **kwargs):
    '''
        Plot a grid of figures such that each subfigure has m subplots.
        mat is a list of lists of 1d data (n, m, x, p)
        where p is the number of plots in each subplot
        titles is a list of titles of length n*m
    '''
    mat = np.array(mat)
    n = len(mat)
    m = len(mat[0])
    mat = mat.reshape(n*m, *mat.shape[2:])
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m*width, n*height))
    for i in tqdm(range(n), desc=desc):
        for j in range(m):
            if i*m+j >= len(mat):
                break
            img = mat[i*m+j]
            #the following is a patch to fix an indexing error in matplotlib
            if n == 1:
                axs = axes[j]
            elif m == 1:
                axs = axes[i]
            else:
                axs = axes[i, j]
            plt.sca(axs)
            if ind:
                plotter(img, ind=i*m+j)
            else:
                plotter(img)
            if titles is not None:
                axs.set_title(titles[i*m+j])
    for key in kwargs:
        eval(f'plt.{key}')(kwargs[key])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
def plot_square_grid1d(mat, titles=None, desc='Grid plot', plotter=plt.plot, width=5, height=1, ind=False, **kwargs):
    '''
        Plot a grid of n subplots.
        mat is a list of lists of 1d data (n, x, p)
        where p is the number of plots in each subplot
        titles is a list of titles of length n
    '''
    mat = np.array(mat)
    if mat.ndim == 2:
        mat = mat[..., None]
    n = int(np.ceil(len(mat)**0.5))
    m = int(np.ceil(len(mat)/n))
    mat = np.pad(mat, ((0, n*m-len(mat)), (0, 0), (0, 0)))
    mat = mat.reshape(n, m, *mat.shape[1:])
    titles = None if titles is None else list(titles) + ['']*(n*m-len(titles))
    return plot_grid1d(mat, titles=titles, desc=desc, plotter=plotter, width=width, height=height, ind=ind, **kwargs)

def plot_square_grid(mat, titles=None, vmin=None, vmax=None, desc='Grid plot', plotter=None, **kwargs):
    '''
        Plot a grid of figures such that each subfigure has m subplots.
        mat is a list of lists of image data (n, m, x, y) or (n*m, x, y)
        titles is a list of titles of length n*m
    '''
    mat = np.array(mat)
    if mat.ndim == 4:
        n = len(mat)
        m = len(mat[0])
        mat = mat.reshape(n*m, *mat.shape[2:])
    else:
        n = int(np.ceil(len(mat)**0.5))
        m = int(np.ceil(len(mat)/n))
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m, n))
    
    for i in tqdm(range(n), desc=desc):
        for j in range(m):
            if i*m+j >= len(mat):
                break
            img = mat[i*m+j]
            #the following is a patch to fix an indexing error in matplotlib
            if n == 1:
                axs = axes[j]
            elif m == 1:
                axs = axes[i]
            else:
                axs = axes[i, j]
            if plotter is not None:
                plt.sca(axs)
                plotter(img)
            else:
                axs.imshow(img, vmin=vmin, vmax=vmax, interpolation='none')
                axs.axis('off')
                if titles is not None:
                    axs.set_title(titles[i*m+j])
            
    for key in kwargs:
        eval(f'plt.{key}')(kwargs[key])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
    
def plot_grid(mat, titles=None, vmin=None, vmax=None, desc='Grid plot', **kwargs):
    '''
        Plot a grid of figures such that each subfigure has m subplots.
        mat is a list of lists of image data (n, m, x, y)
        titles is a list of titles of length n.
    '''
    n = len(mat)
    m = len(mat[0])
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m, n))
    
    for i in tqdm(range(n), desc=desc):
        for j in range(m):
            img = mat[i][j]
            #the following is a patch to fix an indexing error in matplotlib
            if n == 1:
                axs = axes[j]
            elif m == 1:
                axs = axes[i]
            else:
                axs = axes[i, j]
                
            axs.imshow(img, vmin=vmin, vmax=vmax, interpolation='none')
            axs.axis('off')
            if titles is not None:
                axs.set_title(titles[i])
    
    for key in kwargs:
        eval(f'plt.{key}')(kwargs[key])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

#TODO probably move this to project specific repo.
def plot_split_grid(mat, titles=None, vmin=None, vmax=None, desc='Grid plot', **kwargs):
    '''
        Plot a grid of figures such that each subfigure has m subplots.
        mat is a list of lists of image data (n, m, x, y)
        titles is a list of titles of length n.
        colors are split such that first half of m is red and second half is blue.
    '''
    n = len(mat)
    m = len(mat[0])
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m, n))
    
    for i in tqdm(range(n), desc=desc):
        for j in range(m):
            img = mat[i][j]
            #the following is a patch to fix an indexing error in matplotlib
            if n == 1:
                axs = axes[j]
            elif m == 1:
                axs = axes[i]
            else:
                axs = axes[i, j]
            axs.imshow(img, vmin=vmin, vmax=vmax, interpolation='none')
            # axs.axis('off')
            axs.set_xticks([])
            axs.set_yticks([])
            if j >= m//2:
                for spine in axs.spines.values():
                    spine.set_edgecolor('blue')
                    spine.set_linewidth(2)
            else:
                for spine in axs.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            if titles is not None:
                axs.set_title(titles[i])
    
    for key in kwargs:
        eval(f'plt.{key}')(kwargs[key])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def show_stim_movie(stim, path="stim_video.mp4", fps=30, normalizing_constant=None):
    """
        Given stim show an interactive movie.
        
        stim: (time, 1, x, y)
    """
    stim = stim[:, 0].detach().cpu().numpy()
    if normalizing_constant is None:
        stim = stim/np.abs(stim).max()*127 + 127
    else:
        stim = stim * normalizing_constant + 127
    stim = stim.astype(np.uint8)
    writer = imageio.get_writer(path, fps=fps)
    for i in stim:
        writer.append_data(i)
    writer.close()
    w, h = stim.shape[1:]
    return Video(path, embed=True, width=w*3, height=h*3)

def plot_stim(stim, fig=None, title=None, subplot_shape=(1, 1)):
    if fig is None:
        plt.figure()
    if title is not None:
        plt.title(title)
    c = int(np.ceil(np.sqrt(stim.shape[-1])))
    r = int(np.ceil(stim.shape[-1] / c))
    for i in range(stim.shape[-1]):
        ind = (i%c) + (i//c)*c*subplot_shape[1]
        plt.subplot(r*subplot_shape[0],c*subplot_shape[1],ind+1)
        plt.imshow(stim[..., i], vmin=stim.min(), vmax=stim.max())
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    plt.tight_layout()
    return fig

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
    
