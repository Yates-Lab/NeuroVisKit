'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt
import torch
import dill
from utils.utils import seed_everything, unpickle_data, initialize_gaussian_envelope, memory_clear, get_datasets
from utils.schedule import get_train_loop, get_test_objective
from utils.trainer import train
from models import ModelWrapper, CNNdense
seed_everything(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Intended device: ', device)

#%%
'''
    User-Defined Parameters
'''
RUN_NAME = 'test' # Name of log dir.
seed_model = 420
nsamples_train=236452
nsamples_val=56643
overwrite = False

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"n:s:o",["name=", "seed=", "train=", "val="])
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            RUN_NAME = arg
        elif opt in ("-s", "--seed"):
            seed_model = int(arg)
        elif opt == "--train":
            nsamples_train = int(arg)
        elif opt == "--val":
            nsamples_val = int(arg)
        elif opt in ("-o"):
            overwrite = True

filters = [20, 20, 20, 20]
kernels = [11, 11, 11, 11]
#%%
# Prepare helpers for training.
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
checkpoint_dir = os.path.join(dirname, RUN_NAME)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
elif not overwrite:
    print('Directory already exists. Exiting.')
    print('If you want to overwrite, use the -o flag.')
    sys.exit()
    
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids, input_dims, mu, fix_inds = session['cids'], session['input_dims'], session['mu'], session['fix_inds']
fix_inds = fix_inds
NC = len(cids)

def get_model(num_filters, filter_width, device=device):
    seed_everything(seed_model)
    scaffold = [len(num_filters)-1]
    num_inh = [0]*len(num_filters)
    modifiers = None
    cr0 = CNNdense(
            input_dims,
            num_filters,
            filter_width,
            scaffold,
            num_inh,
            is_temporal=False,
            NLtype='relu',
            batch_norm=True,
            norm_type=0,
            noise_sigma=0,
            NC=NC,
            bias=False,
            reg_core=None,
            reg_hidden=None,
            reg_readout={'glocalx':.1, 'l2':0.1},
            reg_vals_feat={'l1':0.01},
            cids=cids,
            modifiers=modifiers,
            window='hamming',
            device=device)

    # initialize parameters
    w_centered = initialize_gaussian_envelope( cr0.core[0].get_weights(to_reshape=False), cr0.core[0].filter_dims)
    cr0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)
    if mu is not None and hasattr(cr0.readout, 'mu'):
        cr0.readout.mu.data = torch.from_numpy(mu[cids].copy().astype('float32')).to(device)
        cr0.readout.mu.requires_grad = True
        cr0.readout.sigma.data.fill_(0.5)
        cr0.readout.sigma.requires_grad = True
    model = ModelWrapper(cr0)
    model.prepare_regularization()
    return model

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
train_dl, val_dl, train_ds, _ = get_datasets(train_data, val_data, device=device)

#%%
memory_clear()
model = get_model(filters, kernels)
max_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

val_loss_min = train(
    model.to(device),
    train_dl,
    val_dl,
    optimizer=optimizer,
    max_epochs=max_epochs,
    verbose=2,
    checkpoint_path=checkpoint_dir,
    device=device,
    patience=30)