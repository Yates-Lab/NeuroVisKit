'''
    Utils for script for generating a nice fitting pipeline.
'''
#%%
import os
import random
from time import sleep
import numpy as np
import torch
import matplotlib
from utils.train import train
from utils.utils import memory_clear, get_datasets, unpickle_data

matplotlib.use('Agg')
#%%
def fig2np(fig):
    '''
        Convert a matplotlib figure to a numpy array.
    '''
    fig.canvas.draw()
    np_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return np_data

# def log_transients(model, train_ds, fix_inds, cids):
#     '''
#         Log transients for a given model, data, and fixational inds.
#     '''
#     fname = os.path.join(os.getcwd(), 'checkpoint', 'transients.png')
#     fig = plot_transients(model, train_ds, fix_inds, cids)
#     fig.savefig(fname)
#     # mat = fig2NP(fig)[None, ...]
#     # file_writer = tf.summary.create_file_writer(os.getcwd())
#     # with file_writer.as_default():
#     #     tf.summary.image("Transients", tf.convert_to_tensor(mat), step=0)
#     plt.close(fig)

def train_loop_org(config,
                   get_model,
                   train_data=None,
                   val_data=None,
                   fixational_inds=None,
                   cids=None,
                   device=None,
                   checkpoint_dir=None,
                   verbose=1,
                   patience=50,
                   seed=None):
    '''
        Train loop for a given config.
    '''
    device = torch.device(device if device else "cuda")
    memory_clear()
    model = get_model(config, device, seed)
    if not train_data or not val_data:
        print("Ray dataset not working. Falling back on pickled dataset.")
        train_data, val_data = unpickle_data(device=device)
    train_dl, val_dl, train_ds, _ = get_datasets(
        train_data, val_data, device=device)
    max_epochs = config['max_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    del model, optimizer
    return {"score": -val_loss_min}

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
        if seed is not None:
            print("Seeded model: ", seed)
        if isinstance(seed, int):
            seed = [seed]*5
        self.seed = seed
        self.lock = Lock()
        self.model_func = model_func
    def get_model(self, config, device="cpu", seed=None):
        with self.lock:
            if seed is not None:
                if isinstance(seed, int):
                    seed = [seed]*5
                self.seed = seed
            if self.seed is not None:
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

# %%
