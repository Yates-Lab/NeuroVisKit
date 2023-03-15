import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.signal import correlate
import skbio
import torch
from skbio.stats.ordination import pcoa as skbio_pcoa
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from . import toy_datasets as tds
import umap as umap_l

def corr_dist(x, y):
    '''
        Compute the distance between two ND arrays using correlation.
    '''
    corr = correlate(x, y, mode='full').flatten()
    mc = corr[np.argmax(np.abs(corr))]
    norm = ((x**2).sum()*(y**2).sum())**0.5
    return 1 if norm == 0 else mc/norm/2 + 0.5

def corr_coef(x, y):
    corr = correlate(x, y, mode='full').flatten()
    mc = corr[np.argmax(np.abs(corr))]
    norm = ((x**2).sum()*(y**2).sum())**0.5
    return mc/norm

def cos_dist(x, y):
    '''
        Compute the cos distance between two ND arrays.
    '''
    x, y = x.ravel(), y.ravel()
    val = x.dot(y) / (x.dot(x) * y.dot(y))**0.5
    return 1 if norm == 0 else val/2 + 0.5

def test_ds(ds, colors, dist_fn=corr_dist, transform_fn=None, plot_ds=True):
    #ds, colors = tds.get_center_surround_ds(100, 35, [[0.1, 0.5, 0.1, 0.3], [0.15, 0.1, 0.5, 0.4]], scale_coeff=[1])
    #ds = tds.add_noise(ds, 0.01 * ds.max())
    if plot_ds:
        tds.plot_ds(ds[:20])
    dist_mat = get_dist_matrix(ds, dist_fn).numpy() # TODO possibly try sqrt of matrix or mat squared
    if transform_fn is not None:
        dist_mat = transform_fn(dist_mat)
    EMBEDDING_DICT = {
        "PCoA": pcoa(dist_mat, n_components=3),
        "PCA": pca(dist_mat, n_components=3),
        "UMAP": umap(dist_mat, n_components=3, n_neighbors=min(15, len(ds)-1))
    }
    for key, comps in EMBEDDING_DICT.items():
        plot_components(comps, colors, title=f'{key}')
    return EMBEDDING_DICT

def hierarchical(dist_mat, colors, optimal_ordering=False):
    # Create hierarchical clustering
    hc = linkage(squareform(dist_mat), optimal_ordering=optimal_ordering)
    # Plot dendrogram
    plt.figure()
    dn = dendrogram(hc, labels=colors)

def MST(dist_mat):
    '''
        Perform MST transformation on dist_mat (i.e. zero out all edges not in the MST)
    '''
    out = minimum_spanning_tree(dist_mat).toarray()
    assert (out + out.T).max() == out.max()
    return out + out.T

def pca(mat, n_components=3, pytorch=False, **kwargs):
    '''
        Perform PCA on mat.
        Return the projected matrix on the first n_components.
    '''
    if pytorch:
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        assert S[0] == S.max()
        return mat @ Vh[:n_components].T
    fit = PCA(n_components=n_components)
    fit.fit_transform(mat)
    return mat @ fit.components_.T

def umap(mat, n_components=2, n_neighbors=3):
    '''
        Perform UMAP on mat.
        Return the projected matrix on the first n_components.
    '''
    fit = umap_l.UMAP(n_components=n_components, n_neighbors=n_neighbors)
    return fit.fit_transform(mat)

def pcoa(dist_mat, eigenv_cutoff=None, n_components=3):
    '''
        Perform PCoA on dist_mat.
        Return the projected matrix on the first n_components.
        Add the min eigenvalue to the diagonal of dist_mat if the min eigenvalue is less than eigenv_cutoff.
    '''
    n_components = -1 if n_components is None else n_components
    if eigenv_cutoff is None:
        out = skbio_pcoa(dist_mat)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = skbio_pcoa(dist_mat)
            min_eigenval = min(out.eigvals.values)
            if min_eigenval < eigenv_cutoff:
                dist_mat = dist_mat + abs(min_eigenval)
                dist_mat.diag().fill(0)
                out = skbio_pcoa(dist_mat)
    return out.samples.values[:, :n_components]
    
def find_most_equidistant(arr, dist_fn):
    '''
        Find the element in arr that is most equidistant from all other elements.
        Return the element and its index.
    '''
    n = len(arr)
    dists = np.zeros(n)
    for i in range(n):
        for j in range(i+1, n):
            dists[i] += dist_fn(arr[i], arr[j])
    return arr[np.argmin(dists)]    

def norm(x):
    return (x - x.mean()) / x.std()

def norm_mean(x):
    return x - x.mean()

def corr_dist_normed(x, y, norm_fn=norm_mean):
    return corr_dist(norm_fn(x), norm_fn(y))

def corr_full(x, y):
    '''
        Compute the correlation between two ND arrays.
    '''
    corr = correlate(x, y, mode='full').sum()
    xsum = correlate(x, x, mode='full').sum()
    ysum = correlate(y, y, mode='full').sum()
    norm = (xsum*ysum)**0.5
    if norm == 0:
        return 0 if x.any() or y.any() else 1
    return corr/norm

def corr_dist_full(x, y):
    '''
        Compute the distance between two ND arrays using correlation.
    '''
    corr = correlate(x, y, mode='full').flatten()
    mc = np.sign(corr.sum()) * np.abs(corr).sum()
    xsum = correlate(x, x, mode='full').sum()
    ysum = correlate(y, y, mode='full').sum()
    norm = (xsum*ysum)**0.5
    return 1 if norm == 0 else mc/norm/2 + 0.5

def get_dist_matrix(arr, dist_fn=corr_dist, pytorch=True):
    '''
        Compute the distance matrix between all elements in arr using dist_fn.
    '''
    n = len(arr)
    if pytorch:
        dists = torch.empty((n, n))
        dists.fill_diagonal_(0)
    else:
        dists = np.empty((n, n))
        np.fill_diagonal(dists, 0)
    for i in range(n):
        for j in range(i+1, n):
            d = dist_fn(arr[i], arr[j])
            dists[i, j] = d
            dists[j, i] = d
    return dists
    
def cluster(arr, n_clusters, n_iter=100, eps=1e-3, dist_fn=corr_dist):
    '''
        Cluster a list of N-d arrays into n_clusters using .
        Return the cluster centers and the cluster assignments.
    '''
    n = len(arr)
    # initialize cluster centers
    arr = np.array(arr)
    centers = np.random.choice(np.arange(len(arr)), n_clusters, replace=False)
    centers = arr[centers]
    for _ in range(n_iter):
        # assign each element to the closest cluster center
        dists = np.full(n, np.inf)
        assignments = np.full(n, np.nan)
        for i in range(n):
            for j in range(n_clusters):
                new_dist = dist_fn(arr[i], centers[j])
                if new_dist < dists[i]:
                    assignments[i] = j
                    dists[i] = new_dist
        # update cluster centers
        new_centers = np.zeros_like(centers)
        for i in range(n_clusters):
            new_centers[i] = find_most_equidistant(arr[assignments == i], dist_fn)
        # check convergence
        if np.all(np.abs(new_centers - centers) < eps):
            print('Converged after {} iterations'.format(_+1))
            break
        centers = new_centers
    return centers, assignments

def plot_components(comps, colors, title='PCoA'):
    zipped = list(zip(*list(np.transpose(comps)), colors))
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=20)
    plt.subplot(1, 3, 1)
    plt.xlabel('PC1')
    plt.title('1D')
    done_colors = []
    for p1, _, _, c in zipped:
        lbl = None
        if c not in done_colors and c is not None:
            lbl = 'rgbcmyk'.index(c)
            done_colors.append(c)
        plt.scatter(p1, 1, c=c, label=lbl)
    plt.gca().set_aspect('equal')
    if len(done_colors) > 0:
        plt.legend()
    plt.subplot(1, 3, 2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D')
    done_colors = []
    for p1, p2, _, c in zipped:
        lbl = None
        if c not in done_colors and c is not None:
            lbl = 'rgbcmyk'.index(c)
            done_colors.append(c)
        plt.scatter(p1, p2, c=c, label=lbl)
    if len(done_colors) > 0:
        plt.legend()
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D')
    done_colors = []
    for p1, p2, p3, c in zipped:
        lbl = None
        if c not in done_colors and c is not None:
            lbl = 'rgbcmyk'.index(c)
            done_colors.append(c)
        ax.scatter(p1, p2, p3, c=c, label=lbl)
    if len(done_colors) > 0:
        plt.legend()
    plt.tight_layout()
    plt.show()