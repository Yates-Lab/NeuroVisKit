
'''
    Preprocess data for training models.
'''
#%% Imports
import os, sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import torch
import dill
from copy import deepcopy
from datasets.mitchell.pixel import Pixel
from datasets.mitchell.pixel.utils import get_stim_list, plot_shifter
from models.utils import plot_stas, eval_model
from models import CNNdense, Shifter
from NDNT.utils import get_max_samples, load_model
from NDNT.training import Trainer, EarlyStopping
from utils.utils import initialize_gaussian_envelope, memory_clear, seed_everything, get_datasets, plot_transients
import utils.preprocess as utils
from matplotlib.backends.backend_pdf import PdfPages

#%% Parameters
'''
    User-Defined Parameters
'''
SESSION_NAME = '20200304'
NUM_LAGS = 24
TRAIN_SHIFTER = True

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"t:s:l:",["train_shifter=", "session=", "lags="])
    for opt, arg in opts:
        if opt in ("-t", "--train_shifter"):
            TRAIN_SHIFTER = arg.upper() == 'TRUE'
        elif opt in ("-s", "--session"):
            SESSION_NAME = arg
        elif opt in ("-l", "--lags"):
            NUM_LAGS = int(arg)

DATADIR = [
    '/Data/stim_movies/',
    '/mnt/Data/Datasets/MitchellV1FreeViewing/stim_movies/'
    ][0]
WINDOW_SIZE = 35
APPLY_SHIFTER = True
TRAIN_FRAC = 0.85
batch_size = 1000
seed = 1234
spike_sorting = 'kilowf'
device = torch.device('cpu')
train_device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
val_device = train_device # if you're cutting it close, put the validation set on the cpu
dtype = torch.float32
sesslist = list(get_stim_list().keys())
assert SESSION_NAME in sesslist, "session name %s is not an available session" %SESSION_NAME
NBname = 'shifter_{}'.format(SESSION_NAME)
cwd = os.getcwd()
NBname = f'shifter_{SESSION_NAME}_{seed}'
dirname = os.path.join(cwd, 'data')
if not os.path.exists(dirname):
    os.makedirs(dirname)
cp_dir = os.path.join(dirname, NBname)
FracDF_include=0.2
#%% Shifter Training
'''
    Training a new shifter.
'''
if TRAIN_SHIFTER:
    ds = utils.get_ds(Pixel, DATADIR, SESSION_NAME, NUM_LAGS, spike_sorting)
    stas, cids, gab_inds = utils.get_ds_analysis(ds, WINDOW_SIZE, FracDF_include)

    maxsamples = get_max_samples(ds, train_device)
    train_data, val_data, train_inds, val_inds = utils.get_data_from_ds(ds, int(TRAIN_FRAC*maxsamples), move_to_cpu=False)

    cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
    cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

    input_dims = ds.dims + [ds.num_lags]

    #Put dataset on GPU
    train_dl, val_dl, _, _ = get_datasets(train_data, val_data, device=train_device, val_device=val_device, batch_size=batch_size)

    def fit_shifter_model(affine=False, overwrite=False):
        seed_everything(seed)
        # manually name the model
        name = 'CNN_shifter'
        if affine:
            name = name + '_affine'
        
        # load best model if it already exists
        exists = os.path.isdir(os.path.join(cp_dir, name))
        if exists and not overwrite:
            try:
                smod = load_model(cp_dir, name)

                smod.to(device)
                val_loss_min = 0
                for data in val_dl:
                    out = smod.validation_step(data)
                    val_loss_min += out['loss'].item()

                val_loss_min/=len(val_dl)    
                return smod, val_loss_min
            except:
                pass

        os.makedirs(cp_dir, exist_ok=True)

        # parameters of architecture
        num_filters = [20, 20, 20, 20]
        filter_width = [11, 11, 11, 11]
        num_inh = [0]*len(num_filters)
        scaffold = [len(num_filters)-1]

        # build CNN
        cr0 = CNNdense(input_dims,
                num_subunits=num_filters,
                filter_width=filter_width,
                num_inh=num_inh,
                cids=cids,
                bias=False,
                scaffold=scaffold,
                is_temporal=False,
                batch_norm=True,
                window='hamming',
                norm_type=0,
                reg_core=None,
                reg_hidden=None,
                reg_readout={'glocalx':1},
                reg_vals_feat={'l1':0.01},
                            )
        
        # initialize parameters
        cr0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,cids].mean(dim=0)) - 1)
        w_centered = initialize_gaussian_envelope( cr0.core[0].get_weights(to_reshape=False), cr0.core[0].filter_dims)
        cr0.core[0].weight.data = torch.tensor(w_centered, dtype=torch.float32)
        
        # build regularization modules
        cr0.prepare_regularization()

        # wrap in a shifter network
        smod = Shifter(cr0, affine=affine)
        smod.name = name

        optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)
        
        # minimal early stopping patience is all we need here
        earlystopping = EarlyStopping(patience=3, verbose=False)

        trainer = Trainer(optimizer=optimizer,
            device = train_device,
            dirpath = os.path.join(cp_dir, smod.name),
            log_activations=False,
            early_stopping=earlystopping,
            verbose=2,
            max_epochs=100)

        # fit and cleanup memory
        memory_clear()
        trainer.fit(smod, train_dl, val_dl)
        val_loss_min = deepcopy(trainer.val_loss_min)
        del trainer
        memory_clear()
        
        return smod, val_loss_min

    # fit shifter with translation only
    mod0, loss0 = fit_shifter_model(affine=False, overwrite=True)

    # fit shifter with affine
    mod1, loss1 = fit_shifter_model(affine=True, overwrite=True)

    ll0 = eval_model(mod0, val_dl)
    ll1 = eval_model(mod1, val_dl)

    # %matplotlib inline
    fig = plt.figure()
    plt.plot(ll0, ll1, '.')
    plt.plot(plt.xlim(), plt.xlim(), 'k')
    plt.title("LL")
    plt.xlabel("Translation shifter")
    plt.ylabel("Affine Shifter")

    plt.show()
    mod1.model.core[0].plot_filters()
    _,fig00 = plot_shifter(mod0.shifter, show=False)
    _,fig01 = plot_shifter(mod1.shifter, show=False)
    # plot STAs before and after shifting
    iix = (train_data['stimid']==0).flatten()
    y = (train_data['robs'][iix,:]*train_data['dfs'][iix,:])/train_data['dfs'][iix,:].sum(dim=0).T
    stas = (train_data['stim'][iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

    _,_,fig02 =  plot_stas(stas.numpy(), title='no shift')

    # do shift correction
    shift = mod0.shifter(train_data['eyepos'])
    stas0 = (mod0.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

    _,_,fig03 =  plot_stas(stas0.detach().numpy(), title='translation')

    shift = mod1.shifter(train_data['eyepos'])
    stas1 = (mod1.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

    _,_,fig04 =  plot_stas(stas1.detach().numpy(), title='affine')

    model = mod0

    sta_true, sta_hat, fig05 = plot_transients(model, val_data)
    filename = os.path.join(dirname, 'shifter_summary_%s_%d.pdf' %(SESSION_NAME, seed))
    p = PdfPages(filename)
    for fig in [fig, fig00, fig01, fig02, fig03, fig04, fig05]: 
        fig.savefig(p, format='pdf') 
        
    p.close()  

    # Save shifter output file
    shifters = [mod0.shifter, mod1.shifter]
    shifter = deepcopy(shifters[np.argmin([loss0, loss1])])

    out = {'cids': cids,
        'shifter': shifter,
        'shifters': shifters,
        'vernum': [0,1],
        'valloss': [loss0, loss1],
        'numlags': NUM_LAGS,
        'tdownsample': 2,
        'eyerad': 8,
        'input_dims': mod0.input_dims,
        'seed': seed}

    fname = 'shifter_' + SESSION_NAME + '_' + ds.spike_sorting + '.p'
    fpath = os.path.join(DATADIR,fname)

    with open(fpath, 'wb') as f:
        dill.dump(out, f)
    del ds, train_data, val_data, train_dl, val_dl, mod0, mod1, out
    memory_clear()
#%% Preprocess
'''
    Preproces.
'''
# Get dataset object from data directory.
ds = utils.get_ds(Pixel, DATADIR, SESSION_NAME, NUM_LAGS, spike_sorting, load_shifters=APPLY_SHIFTER)

# Analyze dataset.
stas, cids, gab_inds = utils.get_ds_analysis(ds, WINDOW_SIZE, FracDF_include)
mu, bestlag, _ = plot_stas(stas.detach().numpy())
f = utils.plot_sacta(ds)

#%% Data storage.
train_data, val_data, train_inds, val_inds = utils.get_data_from_ds(ds)

print('New stim shape {}'.format(train_data['stim'].shape))
indsG = np.where(np.in1d(val_inds, gab_inds))[0]
indsN = np.where(~np.in1d(val_inds, gab_inds))[0]

print('{} training samples, {} validation samples, {} Gabor samples, {} Image samples'.format(len(train_inds), len(val_inds), len(indsG), len(indsN)))

cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

input_dims = ds.dims + [ds.num_lags]
mean_robs = ds.covariates['robs'][:,cids].mean(dim=0)

# Store data and get dict of paths.
with open(os.path.join(dirname, 'metadata.txt'), 'w') as f:
    f.write(f'Session: {SESSION_NAME};\n\
            num_lags: {NUM_LAGS};\n\
            train_shifter: {TRAIN_SHIFTER};\n\
            apply_shifter: {APPLY_SHIFTER};\n\
            window_size: {WINDOW_SIZE};\n\
            frac_df_include: {FracDF_include};\n\
            num_neurons: {len(cids)};\n\
            num_train: {len(train_inds)};\n\
            num_val: {len(val_inds)};\n\
            train_frac: {TRAIN_FRAC};\n')
paths = utils.store_data(train_data, val_data, cwd=cwd)

#%% Get fixation indices.
fix_inds_org = ds.get_fixation_indices(index_valid=True)
sort_inds = np.argsort([max(i) if np.isin(i, train_inds).all() else np.inf for i in fix_inds_org])
fix_inds_sorted = [fix_inds_org[i] for i in sort_inds]
fix_inds_t = fix_inds_sorted[:len(train_inds)]

#%% Store session info.
session = {
    'fix_inds': fix_inds_t,
    'cids': cids,
    'mu': mu.copy(),
    'input_dims': input_dims
}
with open(os.path.join(paths['data'], 'session.pkl'), 'wb') as f:
    dill.dump(session, f)
#%%