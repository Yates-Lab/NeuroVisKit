'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt
import torch
import dill
import ray
from ray import tune, air
from ray.tune.search.ax import AxSearch
from ray.air.checkpoint import Checkpoint
from ray.air import session
from utils.utils import seed_everything, unpickle_data
from utils.schedule import ModelGenerator, get_train_loop, get_test_objective
from utils.get_models import get_model
init_seed = 0
seed_everything(init_seed)

intended_device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
print('Intended device: ', intended_device)

# %load_ext autoreload
# %autoreload 2

#%%
'''
    User-Defined Parameters
'''
test = True # Short test or full run.
RUN_NAME = 'shifter_run1' # Name of log dir.
seed_model = 420
nsamples_train=236452
nsamples_val=56643

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"t:n:s:",["test=", "name=", "seed=", "train=", "val="])
    for opt, arg in opts:
        if opt in ("-t", "--test"):
            test = arg.upper() == 'TRUE'
        elif opt in ("-n", "--name"):
            RUN_NAME = arg
        elif opt in ("-s", "--seed"):
            seed_model = int(arg)
        elif opt == "--train":
            nsamples_train = int(arg)
        elif opt == "--val":
            nsamples_val = int(arg)

modify_tmp_dir = False # If low on space, can modify tmp dir to a different drive. This requires starting ray from command line to indicate the new tmp dir.

#%%
# Prepare helpers for training.
if not test and modify_tmp_dir:
    ray.init(address='auto')
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
logdir = os.path.join(dirname, 'tensorboard')
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
    
config = {
    'model': 'CNNdense',
    'cids': session['cids'],
    'input_dims': session['input_dims'],
    'mu': session['mu'],
    'fix_inds': session['fix_inds'],
    'seed': seed_model,
}
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

model_gen = get_model(config, factory=True)
train_loop = get_train_loop(model_gen, session, Checkpoint)
test_objective = get_test_objective(model_gen, session, Checkpoint, intended_device)

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)

#%%
# Run.
if test:
    test_objective(train_data, val_data)
else:
    trainable = tune.with_parameters(
        train_loop, t_data=train_data, v_data=val_data)
    trainable = tune.with_resources(
        trainable, {'gpu': 1, 'cpu': 8})

    print('Creating tuner.')
    axopt = AxSearch(metric='score', mode='max')
    tuner = tune.Tuner(trainable,
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

