import torch
from utils.trainer import train
from utils.utils import seed_everything, initialize_gaussian_envelope
from models import ModelWrapper, CNNdense

def train_lbfgs(model, train_loader, val_loader, checkpoint_dir, device=None):
    pass

def train_adam(model, train_loader, val_loader, checkpoint_dir, device):
    max_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    val_loss_min = train(
        model.to(device),
        train_loader,
        val_loader,
        optimizer=optimizer,
        max_epochs=max_epochs,
        verbose=2,
        checkpoint_path=checkpoint_dir,
        device=device,
        patience=30)
    return val_loss_min

TRAINER_DICT = {
    'adam': train_adam,
    'lbfgs': train_lbfgs,
}

def get_CNNdense(config, seed, cids, input_dims, device):
    num_filters = config['filters']
    filter_width = config['kernels']
    mu = config['mu']
    
    seed_everything(seed)
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
            NC=len(cids),
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

MODEL_DICT = {
    'CNNdense': get_CNNdense,
}