'''
    Utils for script for generating a nice fitting pipeline.
'''
#%%
import os
import gc
import random
from time import sleep
from copy import deepcopy
import numpy as np
import torch
import dill
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from datasets.generic import GenericDataset
from models.base import Encoder
from train import train

matplotlib.use('Agg')

def fig2np(fig):
    '''
        Convert a matplotlib figure to a numpy array.
    '''
    fig.canvas.draw()
    np_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return np_data


def plot_transients(model, train_ds, fix_inds, cids):
    '''
        Plot transients for a given model, data, and fixational inds.
    '''
    nfix = len(fix_inds)
    ncids = len(cids)
    nbins = 120
    fig = plt.figure()
    org_device = train_ds.device

    plt.subplot(2, 1, 1)
    robs = torch.full((nfix, nbins, ncids), float('nan'), device='cpu')
    for ifix in range(nfix):
        items = min(len(fix_inds[ifix]), nbins)
        data = train_ds[fix_inds[ifix]]
        robs[ifix, :items, :] = data['robs'][:items, cids].detach().clone().cpu()
    print(np.prod(robs.shape).item()-torch.isnan(robs).sum())
    for cid_ind in range(ncids):
        plt.plot(np.nanmean(robs[:, :, cid_ind], axis=0))
    del robs
    # memory_clear()
    plt.subplot(2, 1, 2)
    robs = float('nan')*torch.ones((nfix, nbins, ncids))
    with torch.no_grad():
        for ifix in range(nfix):
            items = min(len(fix_inds[ifix]), nbins)
            data = train_ds[fix_inds[ifix]]
            data_dict = {}
            for key, val in data.items():
                data_dict[key] = val[:items, ...].cpu()
            robs[ifix, :items, :] = model(data_dict)[:items, :]
            for key, val in data.items():
                val[:items, ...].to(org_device)
    print(np.prod(robs.shape).item()-torch.isnan(robs).sum())
    for cid_ind in range(ncids):
        plt.plot(np.nanmean(robs[:, :, cid_ind], axis=0))
    del robs
    # memory_clear()
    return fig


def log_transients(model, train_ds, fix_inds, cids):
    '''
        Log transients for a given model, data, and fixational inds.
    '''
    fname = os.path.join(os.getcwd(), 'checkpoint', 'transients.png')
    fig = plot_transients(model, train_ds, fix_inds, cids)
    fig.savefig(fname)
    # mat = fig2NP(fig)[None, ...]
    # file_writer = tf.summary.create_file_writer(os.getcwd())
    # with file_writer.as_default():
    #     tf.summary.image("Transients", tf.convert_to_tensor(mat), step=0)
    plt.close(fig)


def unpickle_data(nsamples_train=None, nsamples_val=None, device="cpu"):
    '''
        Get training and validation data from pickled files and place on device.
    '''
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'data', 'train')
    train_data_local = {}
    num_samples = [None, None]
    for file in os.listdir(train_path):
        loaded = torch.load(os.path.join(train_path, file), map_location=device)
        num_samples[0] = loaded.shape[0]
        train_data_local[file[:-3]
                         ] = loaded[:nsamples_train].clone()
        del loaded
        memory_clear()
    val_path = os.path.join(cwd, 'data', 'val')
    val_data_local = {}
    for file in os.listdir(val_path):
        loaded = torch.load(os.path.join(val_path, file), map_location=device)
        num_samples[1] = loaded.shape[0]
        val_data_local[file[:-3]
                       ] = loaded[:nsamples_val].clone()
        del loaded
        memory_clear()
    print(f'Loaded {nsamples_train} training samples and {nsamples_val} validation samples')
    print(f'Out of {num_samples[0]} training samples and {num_samples[1]} validation samples')
    return train_data_local, val_data_local


def get_datasets(train_data, val_data, device=None, batch_size=1000):
    '''
        Get datasets from data files.
    '''
    train_ds = GenericDataset(train_data, device)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)
    val_ds = GenericDataset(val_data, device)
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    return train_dl, val_dl, train_ds, val_ds


def memory_clear():
    '''
        Clear unneeded memory.
    '''
    torch.cuda.empty_cache()
    gc.collect()


def train_loop_org(config,
                   get_model,
                   train_data=None,
                   val_data=None,
                   fixational_inds=None,
                   cids=None,
                   device=None,
                   checkpoint_dir=None,
                   verbose=1,
                   patience=20):
    '''
        Train loop for a given config.
    '''
    device = torch.device(device if device else "cuda")
    model = get_model(config, device)
    memory_clear()
    if not train_data or not val_data:
        print("Ray dataset not working. Falling back on pickled dataset.")
        train_data, val_data = unpickle_data(device=device)
    train_dl, val_dl, train_ds, _ = get_datasets(
        train_data, val_data, device=device)
    memory_clear()
    max_epochs = config['max_epochs']
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-3)
    val_loss_min = train(
        model.to(device),
        train_dl,
        val_dl,
        optimizer=optimizer,
        max_epochs=max_epochs,
        verbose=verbose,
        checkpoint_path=checkpoint_dir,
        device=device,
        patience=patience)
    # if fixational_inds is not None and cids is not None:
    #     log_transients(model, train_ds, fixational_inds, cids)
    return {"score": -val_loss_min}

def load_model(checkpoint_path, model):
    '''
        Load model from checkpoint of trainer, if using state dict.
    '''
    ckpt = dill.load(open(os.path.join(checkpoint_path, 'state.pkl'), 'rb'))
    model.load_state_dict(ckpt['net'])
    epoch = ckpt['epoch']
    print(f"Loaded model from checkpoint. {epoch} epochs trained.")
    return model

class Lock:
    def __init__(self):
        self.lock = 0
    def __enter__(self):
        while self.get_lock():
            sleep(0.1)
        self.lock_in()
        if self.get_lock() > 1:
            raise Exception("Lock error")
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock -= 1
    def get_lock(self):
        return self.lock
    def lock_in(self):
        self.lock += 1
        
class ModelGenerator:
    '''
        Asynchronously generate models that all have the exact same seeds/initializations.
    '''
    def __init__(self, model_func, seed=0):
        if isinstance(seed, int):
            seed = [seed]*5
        self.seed = seed
        self.lock = Lock()
        self.model_func = model_func
    def get_model(self, config, device="cpu"):
        with self.lock:
            np.random.seed(self.seed[0])
            random.seed(self.seed[1])
            torch.manual_seed(self.seed[2])
            os.environ['PYTHONHASHSEED']=str(self.seed[3])
            torch.cuda.manual_seed(self.seed[4])
            model = self.model_func(config, device)
            return model
    def test(self):
        config_i = {
            **{f"filter_width{i}": val for i, val in enumerate([4, 15, 8, 5])},
            **{f"num_filters{i}": val for i, val in enumerate([17, 22, 22, 28])},
            "num_layers": 4,
            "max_epochs": 90,
            "d2x": 0.00080642,
            "d2t": 0.0013630,
            "center": 0.00013104,
        }
        m1 = self.get_model(config_i)
        m2 = self.get_model(config_i)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                print("!ModelGenerator is not being reproducible!")
                return False
        print("ModelGenerator is being reproducible")
        return True

class base(Encoder):

    def __init__(self,
        input_dims,
        cids,
        modifers):

        super().__init__(input_dims, cids=cids, modifiers=modifers)

        self.core = None
        self.readout = None
        self.name = 'base'
        self.bias = torch.nn.Parameter(torch.zeros(len(cids)))
        
    def forward(self, data):

        offset = self.offval
    
        for offmod,stim in zip(self.offsets, self.offsetstims):
            offset = offset + offmod(data[stim])

        offset += self.bias

        return self.output_NL(offset)
    
    def compute_reg_loss(self):
        return self.offval

def initialize_gaussian_envelope( ws, w_shape):
    """
    This assumes a set of filters is passed in, and windows by Gaussian along each non-singleton dimension
    ws is all filters (ndims x nfilters)
    wshape is individual filter shape
    """
    ndims, nfilt = ws.shape
    assert np.prod(w_shape) == ndims
    wx = np.reshape(deepcopy(ws), w_shape + [nfilt])
    for dd in range(1,len(w_shape)):
        if w_shape[dd] > 1:
            L = w_shape[dd]
            if dd == len(w_shape)-1:
                genv = np.exp(-(np.arange(L))**2/(2*(L/6)**2))
            else:
                genv = np.exp(-(np.arange(L)-L/2)**2/(2*(L/6)**2))

            if dd == 0:
                wx = np.einsum('abcde, a->abcde', wx, genv)
            elif dd == 1:
                wx = np.einsum('abcde, b->abcde', wx, genv)
            elif dd == 2:
                wx = np.einsum('abcde, c->abcde', wx, genv)
            else:
                wx = np.einsum('abcde, d->abcde', wx, genv)
    return np.reshape(wx, [-1, nfilt])

def plot_layer(layer, ind=None):
    
    if layer.filter_dims[-1] > 1:
        ws = layer.get_weights()
        if ind is not None: 
            ws = ws[..., ind]
        ws = np.transpose(ws, (2,0,1,3))
        layer.plot_filters()
    else:
        ws = layer.get_weights()
        if ind is not None: 
            ws = ws[..., ind]

        nout = ws.shape[-1]
        nin = ws.shape[0]
        plt.figure(figsize=(10, 10))

        for i in range(nin):
            for j in range(nout):
                plt.subplot(nin, nout, i*nout + j + 1)
                plt.imshow(ws[i, :,:, j], aspect='auto', interpolation='none')
                plt.axis("off")

def plot_dense_readout(layer):
    ws = layer.get_weights()
    n = ws.shape[-1]
    sx = int(np.ceil(np.sqrt(n)))
    sy = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(sx*2, sy*2))
    for cc in range(n):
        plt.subplot(sx, sy, cc + 1)
        v = np.max(np.abs(ws[:,:,cc]))
        plt.imshow(ws[:,:,cc], interpolation='none', cmap=plt.cm.coolwarm, vmin=-v, vmax=v)

def plot_model(model):

    for layer in model.core:
        plot_layer(layer)
        plt.show()

    if hasattr(model, 'offsets'):
        for i,layer in enumerate(model.offsets):
            _ = plt.plot(layer.get_weights())
            plt.title("Offset {}".format(model.offsetstims[i]))
            plt.show()
    
    if hasattr(model, 'gains'):
        for i,layer in enumerate(model.gains):
            _ = plt.plot(layer.get_weights())
            plt.title("Gain {}".format(model.gainstims[i]))
            plt.show()

    if hasattr(model.readout, 'mu'):
        plt.imshow(model.readout.get_weights())
    else:
        plot_dense_readout(model.readout.space)
        plt.show()
        plt.imshow(model.readout.feature.get_weights())
        plt.xlabel("Neuron ID")
        plt.ylabel("Feature ID")