#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_noise(N, type='white'):
    '''
        Generate 2D noise with dim (N, N).
    '''
    F = np.abs(np.fft.fftfreq(N))
    S = np.ones(F.shape)
    nz = F != 0
    if type == 'pink':
        S[nz] = 1/F[nz]
    elif type == 'brownian':
        S[nz] = 1/F[nz]**2
    elif type == 'blue':
        S = np.sqrt(F)
    elif type == 'violet':
        S = F
    elif type != 'white':
        raise ValueError('Invalid noise type')
    S = S[:, None] @ S[None, :]
    S = S / np.sqrt(np.mean(S**2))
    X_white = np.fft.fftn(np.random.randn(N, N))
    return np.abs(np.fft.ifftn(X_white * S))

def plot_noise(n):
    '''
        Plot all available types of noise.
    '''
    colors = ['white', 'pink', 'brownian', 'blue', 'violet']
    plt.figure()
    for c in range(len(colors)):
        noise = generate_noise(n, colors[c])
        plt.subplot(2, 3, c+1)
        plt.title(colors[c])
        plt.imshow(noise)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        print(colors[c], (noise.min(), noise.max()))
        print("mean", np.mean(noise))
    plt.tight_layout()

def get_mei_start(N, lags, type='white'):
    '''
        Generate a 3D noise with dim (N, N, lags)
    '''
    noise = [generate_noise(N, type) for _ in range(lags)]
    return np.array(noise).transpose((1, 2, 0))

def get_init_dict(dims, mn, mx, std, mean, dl):
    '''
        Generate a dictionary of possible initializations for MEI.
    '''
    def get_init_noise(noise_type):
        init_noise = get_mei_start(*dims[-2:], noise_type)
        init_noise = (init_noise - np.mean(init_noise)) / np.std(init_noise)
        return init_noise * std + mean
    init_zero = np.zeros(dims[-3:])
    init_max = np.full(dims[-3:], mx)
    init_min = np.full(dims[-3:], mn)
    return {
        "zero": init_zero,
        "max": init_max,
        "min": init_min,
        "stim": next(dl)["stim"].reshape(dims[-3:]).detach().numpy(),
        **{noise: get_init_noise(noise) for noise in ["white", "brownian", "pink", "blue", "violet"]}
    }

def mei(model, cids, start, alpha=0.1, nsteps=100, scalefunc=lambda x: 1, eps=1e-3, name=''):
    '''
        Find an MEI for a given model and neuron ids.

        -model: a pytorch model
        -cids: a list of neuron ids
        -start: the initial input
        -alpha: the learning rate
        -nsteps: the number of steps to run
        -scalefunc: scales the learning rate as a func of step
        -eps: the convergence threshold
        -name: name for printing
    '''
    for i in range(nsteps):
        model.zero_grad()
        start_copy = start.clone()
        # Get the current prediction
        pred = model({"stim": start})
        #identify relevant neurons
        out = pred[cids].mean()
        # Get the gradient of the prediction with respect to the input
        grad = torch.autograd.grad(out, start, retain_graph=True)[0]
        if grad is None or grad.sum().item() == 0:
            print("No gradient at step", i)
            return None
        # Update the input
        start = start_copy + scalefunc(i) * alpha * grad
        if torch.abs(start-start_copy).max() < eps:
            print(name, "converged at step", i)
            break
    return start

def get_gratings(dims, n=2, ppd=1/0.025, fps=120):
    '''
        Generate spatio-temporal sinusodial gratings.

        -dims (x, y, t)
        -n: number of frequencies to sample per dimension (x, y, t)
        -frequencies (fx, fy, ft) cycles per unit

        To get the gratings for Fx, Fy, Ft, use the index I as follows:
        -gratings[Ix, Iy, It, :, :, :]
        Where I goes from 0 (F=0) to n-1 (max F, half a cycle/step)
    '''
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    t = np.arange(dims[2])
    fx = np.linspace(0, 0.5, n)
    fy = np.linspace(0, 0.5, n)
    ft = np.linspace(0, 0.5, n)
    FX, FY, FT, X, Y, T = np.meshgrid(fx,fy,ft,x,y,t)
    out_freq = np.array(np.meshgrid(fx*ppd,fy*ppd,ft*fps)).transpose((1, 2, 3, 0))
    gratings = np.cos(2*np.pi*FX*X)*np.cos(2*np.pi*FY*Y)*np.cos(2*np.pi*FT*T)
    return out_freq, gratings

def invariant_mei(model, cids, start, alpha=0.1, nsteps=100, scalefunc=lambda x: 1, eps=1e-3, name='', memory=1, pchange=0.5):
    '''
        Scale along an invariant direction of a previously generated MEI.

        -model: a pytorch model
        -cids: a list of neuron ids
        -start: the initial input (should be an MEI)
        -alpha: the learning rate
        -nsteps: the number of steps to run
        -scalefunc: scales the learning rate as a func of step
        -eps: the convergence threshold
        -name: name for printing/plotting
        -memory: number of previous checkpoints to use for aligning the gradient
        -pchange: threshold for taking a checkpoint
    '''
    checkpoints = [(start.clone(), -1)]
    for step in range(nsteps):
        model.zero_grad()
        start_copy = start.clone()
        # Get the current prediction
        pred = model({"stim": start})
        #identify relevant neurons
        out = pred[cids].mean()
        # Get the gradient of the prediction with respect to the input
        grad = torch.autograd.grad(out, start, retain_graph=True)[0]
        if grad is None or grad.sum().item() == 0:
            print("No gradient at step", step, "(None:", grad is None, ")")
            return None
        # Update the input
        start_intermediate = start + scalefunc(step) * alpha * grad
        pred = model({"stim": start})
        cp = ((start_copy.clone()-start_intermediate)**2).sum()
        for i in range(min(memory, len(checkpoints))):
            cp += ((start_intermediate-checkpoints[-i-1][0])**2).sum()
        out = pred[cids].mean() + cp / (cp**2).sum()
        grad = torch.autograd.grad(out, start, retain_graph=True)[0]
        if grad is None or grad.sum().item() == 0:
            print("No gradient at step", i)
            return None
        start = start_copy + scalefunc(step) * alpha * grad
        diff = torch.abs(start-checkpoints[-1][0])
        if torch.abs(diff).sum()/torch.abs(checkpoints[-1][0]).sum() > pchange:
            print("checkpointed at ", step)
            checkpoints.append((start.clone(), step))
        if torch.abs(start-start_copy).max() < eps:
            print(name, "converged at step", step)
            break
    # if len(checkpoints) > 1:
    #     print(name, "checkpoints:", len(checkpoints)-1)
    #     plt.figure(figsize=(5, 5*len(checkpoints)))
    #     plt.title(name+' - checkpoints')
    #     for i in range(len(checkpoints)):
    #         cp, ind = checkpoints[i]
    #         cp = cp.detach().numpy()
    #         mags = [(cp[..., j]**2).sum() for j in range(cp.shape[-1])]
    #         best_slices = list({np.argmin(mags), np.argmax(mags)})
    #         best_slices.sort()
    #         plt.subplot(len(checkpoints), 2, 2*i+1)
    #         plt.title("step "+str(ind))
    #         plt.imshow(cp[..., best_slices[0]])
    #         plt.gca().set_xticks([])
    #         plt.gca().set_yticks([])
    #         if len(best_slices) > 1:
    #             plt.subplot(len(checkpoints), 2, 2*i+2)
    #             plt.imshow(cp[..., best_slices[1]])
    #             plt.gca().set_xticks([])
    #             plt.gca().set_yticks([])
    #     plt.tight_layout()
    if len(checkpoints) > 1:
        print(name, "checkpoints:", len(checkpoints)-1)
        for i in range(len(checkpoints)):
            cp, ind = checkpoints[i]
            cp = cp.detach().numpy()
            plt.figure()
            plt.suptitle(f"{name} CP - step {ind}")
            c = int(np.ceil(np.sqrt(cp.shape[-1])))
            r = int(np.ceil(cp.shape[-1] / c))
            for i in range(cp.shape[-1]):
                plt.subplot(r,c,i+1)
                plt.imshow(cp[..., i], vmin=cp.min(), vmax=cp.max())
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            plt.tight_layout()
    print("Done")
    return start
# %%
