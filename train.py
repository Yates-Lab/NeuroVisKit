'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, get_datasets
import utils.train as utils
from utils.loss import get_loss
from utils.get_models import get_model
import torch.nn.functional as F
import torch.nn as nn
seed_everything(0)

run_name = 'test' # Name of log dir.
session_name = '20200304'
nsamples_train=236452
nsamples_val=56643
overwrite = False
from_checkpoint = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 420
config = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'CNNdense', # utils/get_models.py for more model options
    'trainer': 'adam', # utils/train.py for more trainer options
    'filters': [32, 16, 32, 64], # the config is fed into the model constructor
    'kernels': [9, 5, 5, 3],
    'preprocess': 'binarize', # this further preprocesses the data before training
    'override_output_NL': True, # this overrides the output nonlinearity of the model according to the loss
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite", "device=", "session=", "loss=", "model=", "trainer=", "preprocess=", "override_NL"]
    opts, args = getopt.getopt(argv,"n:s:oc:d:l:m:t:p:", opt_names)
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            run_name = arg
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-c", "--config"):
            config = config.update(json.loads(arg))
        elif opt == "--train":
            nsamples_train = int(arg)
        elif opt == "--val":
            nsamples_val = int(arg)
        elif opt in ("-o", "--overwrite"):
            overwrite = True
        elif opt in ("--from_checkpoint"):
            from_checkpoint = True
        elif opt in ("-d", "--device"):
            device = torch.device(arg)
        elif opt in ("--session"):
            session_name = arg
        elif opt in ("-l", "--loss"):
            config["loss"] = arg
        elif opt in ("-m", "--model"):
            config["model"] = arg
        elif opt in ("-t", "--trainer"):
            config["trainer"] = arg
        elif opt in ("-p", "--preprocess"):
            config["preprocess"] = arg
        elif opt in ("--override_NL"):
            config["override_output_NL"] = True
            
#%%
# Prepare helpers for training.
config_original = config.copy()
config['device'] = device
print('Device: ', device)
config['seed'] = seed
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
data_dir = os.path.join(dirname, 'sessions', session_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
elif not overwrite and not from_checkpoint:
    print('Directory already exists. Exiting.')
    print('If you want to overwrite, use the -o flag.')
    sys.exit()
    
with open(os.path.join(dirname, 'sessions', session_name, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
config.update({
    'cids': session['cids'],
    'input_dims': session['input_dims'],
    'mu': session['mu'],
})

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device)

#%%
# Load model and preprocess data.
if from_checkpoint:
    with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
        model = dill.load(f).to(device)
else:
    model = get_model(config)

model.loss, nonlinearity = get_loss(config)
if config['override_output_NL']:
    model.model.output_NL = nonlinearity

if config['preprocess'] == 'binarize': # Binarize the data to eliminate r > 1.
    inds2 = torch.where(train_data['robs'] > 1)[0]
    train_data['robs'][inds2 - 1] = 1
    train_data['robs'][inds2 + 1] = 1
    train_data['robs'][inds2] = 1
    inds2 = torch.where(val_data['robs'] > 1)[0]
    val_data['robs'][inds2 - 1] = 1
    val_data['robs'][inds2 + 1] = 1
    val_data['robs'][inds2] = 1
elif config['preprocess'] == 'smooth':
    train_data['robs'] = utils.smooth_robs(train_data['robs'], smoothN=10)
    val_data['robs'] = utils.smooth_robs(val_data['robs'], smoothN=10)
elif config['preprocess'] == 'zscore':
    train_data['robs'] = utils.zscore_robs(utils.smooth_robs(train_data['robs'], smoothN=10))
    val_data['robs'] = utils.zscore_robs(utils.smooth_robs(val_data['robs'], smoothN=10))

trainer = utils.get_trainer(config)
best_val_loss = trainer(model, train_dl, val_dl, checkpoint_dir, device)
#save metadata
with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
    to_write = {
        'run_name': run_name,
        'session_name': session_name,
        'nsamples_train': nsamples_train,
        'nsamples_val': nsamples_val,
        'seed': seed,
        'config': config_original,
        'device': str(device),
        'best_val_loss': best_val_loss,
    }
    f.write(json.dumps(to_write, indent=2))
with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as f:
    dill.dump(config_original, f, 'wb')