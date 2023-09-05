import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def add_noise(ds, noise_std):
    return np.clip(ds + np.random.normal(0, noise_std, size=ds.shape), ds.min(), ds.max())

def plot_ds(ds):
    r = int(np.ceil(len(ds)**0.5))
    c = int(np.ceil(len(ds)/r))
    plt.figure(figsize=(5*c, 5*r))
    for i in range(len(ds)):
        plt.subplot(r, c, i+1)
        plt.imshow(ds[i])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    plt.suptitle('Dataset')
    plt.tight_layout()
    plt.show()

def get_line_ds(n):
    arrs = []
    colors = []
    for j in range(-n//2, n//2):
        a = np.zeros((n, n))
        a[:, n//2 + j] = 1
        arrs.append(a)
        colors.append('r')
        a = np.zeros((n, n))
        a[n//2+j, :] = 1
        arrs.append(a)
        colors.append('b')

    num_seeds = len(arrs)
    for i in range(num_seeds):
        for j in range(10, 45, 10):
            arrs.append(rotate(arrs[i], angle=j, reshape=False))
            arrs.append(rotate(arrs[i], angle=-j, reshape=False))
            colors += ['b', 'b'] if i % 2 == 1 else ['r', 'r']
    return arrs, colors

def get_center_surround_ds(n, size, stds, scale_coeff=[1], sign_flip=False):
    '''
        Generates a dataset of center-surround stimuli.
        n: number of stimuli
        size: size of each stimulus (size x size)
        stds: shape (2, _) array of standard deviations for the center and surround
    '''
    def difference_of_gaussians(mu, sigma1, sigma2):
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        d2 = (x[None, ...]-mu[:, 0])**2 + (y[None, ...]-mu[:, 1])**2
        g1 = sigma2*np.exp(-(d2 / ( 2.0 * sigma1**2 ) ) )
        g2 = sigma1*np.exp(-(d2 / ( 2.0 * sigma2**2 ) ) )
        return (g1 - g2)/np.sqrt(2*np.pi)

    mu_options = np.linspace(-1, 1, size)
    mu_options = mu_options[(mu_options<=0.5) & (mu_options>=-0.5)]
    mus = np.random.choice(mu_options, (n, 2, 1, 1))
    signs = np.random.choice([-1, 1], n) if sign_flip else 1
    sizes = np.random.choice(scale_coeff, n) * signs
    stds = np.array(stds)
    sigmas = np.random.randint(stds.shape[1], size=n)
    color = ['rgbcmyk'[s] for s in sigmas]
    return difference_of_gaussians(mus, *stds[:, sigmas, None, None]) * sizes[:, None, None], color