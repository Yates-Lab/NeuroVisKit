'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os
import random
import torch
import dill
import numpy as np
import ray
from ray import tune, air
from ray.tune.search.ax import AxSearch
from ray.air.checkpoint import Checkpoint
from ray.air import session
import matplotlib
from datasets.mitchell.pixel.utils import get_stim_list
from models import CNNdense
from utils import train_loop_org, unpickle_data, ModelGenerator, initialize_gaussian_envelope, seed_everything
init_seed = 0
seed_everything(init_seed)

intended_device = torch.device(
    'cuda:1' if torch.cuda.is_available() else 'cpu')
print('Intended device: ', intended_device)

%load_ext autoreload
%autoreload 2

#%%
'''
    User-Defined Parameters
'''
test = True # Short test or full run.
modify_tmp_dir = False # If low on space, can modify tmp dir to a different drive. This requires starting ray from command line to indicate the new tmp dir.
RUN_NAME = 'foundation_run' # Name of log dir.
seed_model = 420
nsamples_train=236452
nsamples_val=56643

#%%
# Prepare helpers for training.
if not test and modify_tmp_dir:
    ray.init(address='auto')
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
logdir = os.path.join(dirname, 'tensorboard')
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids, input_dims, mu, fix_inds = session['cids'], session['input_dims'], session['mu'], session['fix_inds']
fix_inds = fix_inds
NC = len(cids)
search_space = {
    **{f'num_filters{i}': tune.qrandint(1, 32, 2) for i in range(4)},
    **{f'filter_width{i}': tune.qrandint(3, 17, 2) for i in range(4)},
    'num_layers': tune.randint(3, 5),
    'max_epochs': tune.qrandint(50, 201, 25),
    # 'd2x': tune.qloguniform(1e-4, 1e-1, 5e-5),
    'd2t': tune.qloguniform(1e-4, 1e-1, 5e-5),
    'center': tune.qloguniform(1e-4, 1e-1, 5e-5),
    'edge_t': tune.qloguniform(1e-4, 1e-1, 5e-5),
}

def get_model(config, device='cpu', dense=True):
    '''
        Create new model.
    '''
    num_layers = config['num_layers']
    num_filters = [config[f'num_filters{i}'] for i in range(num_layers)]
    filter_width = [config[f'filter_width{i}'] for i in range(num_layers)]
    # d2x = config['d2x']
    d2t = config['d2t']
    center = config['center']
    edge_t = config['edge_t']
    scaffold = [len(num_filters)-1]
    num_inh = [0]*len(num_filters)
    # modifiers = {
    #     'stimlist': ['frame_tent', 'fixation_onset'],
    #     'gain': [deepcopy(drift_layer), deepcopy(sac_layer)],
    #     'offset': [deepcopy(drift_layer), deepcopy(sac_layer)],
    #     'stage': 'readout',
    # }

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
    cr0.prepare_regularization()
    return cr0

model_gen = ModelGenerator(get_model, seed_model)

def train_loop(config, t_data=None, v_data=None):
    '''
        Train loop for a given config with checkpointing.
    '''
    cp_dir = os.path.join(os.getcwd(), 'checkpoint')
    os.mkdir(cp_dir)
    out = train_loop_org(config, model_gen.get_model, t_data, v_data,
                         fixational_inds=fix_inds, cids=cids, checkpoint_dir=cp_dir)
    session.report(out, checkpoint=Checkpoint.from_directory(cp_dir))
    return out

def test_objective(t_data, v_data, seed=None, name='checkpoint'):
    '''
        Test objective function for training.
    '''
    filts = [20, 20, 20, 20]
    config_i = {
        **{f'num_filters{i}': filts[i] for i in range(4)},
        **{f'filter_width{i}': 11 for i in range(4)},
        'num_layers': 4,
        'max_epochs': 50,
        'd2x': 0.000001,
        'd2t': 1e-2,
        'center': 1e-2,
        'edge_t': 1e-1,
    }
    cp_dir = os.path.join(os.getcwd(), 'data', name)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    print(train_loop_org(config_i, model_gen.get_model, t_data, v_data,
                         fixational_inds=fix_inds, cids=cids, device=intended_device, checkpoint_dir=cp_dir, verbose=2, seed=seed))

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)

#%%
# Run.
if test:
    try:
        for i in range(100):
            test_objective(train_data, val_data, seed=i, name=f'checkpoint_{i}')
    except Exception as e:
        print(e)
if not test:
    trainable_with_data = tune.with_parameters(
        train_loop, t_data=train_data, v_data=val_data)
    trainable_with_gpu = tune.with_resources(
        trainable_with_data, {'gpu': 1, 'cpu': 8})

    print('Creating tuner.')
    axopt = AxSearch(metric='score', mode='max')
    tuner = tune.Tuner(trainable_with_gpu,
                    tune_config=tune.TuneConfig(
                        search_alg=axopt,
                        max_concurrent_trials=2,
                        num_samples=1000,
                    ),
                    param_space=search_space,
                    run_config=air.RunConfig(
                        log_to_file=True,
                        local_dir=logdir,
                        name=RUN_NAME,
                        # checkpoint_config=air.CheckpointConfig(num_to_keep=5)
                    ))
    print('Starting tuning.')
    results = tuner.fit()
    print(results)

# %%
