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
nsamples_train=None
nsamples_val=None
overwrite = False
from_checkpoint = False
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
seed = 420
config = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'CNNdense', # utils/get_models.py for more model options
    'trainer': 'adam', # utils/train.py for more trainer options
    'filters': [20, 20, 20, 20], # the config is fed into the model constructor
    'kernels': [11, 11, 11, 11],
    'preprocess': [],# this further preprocesses the data before training
    'override_output_NL': False, # this overrides the output nonlinearity of the model according to the loss,
    'pretrained_core': None,
    'defrost': False,
    'batch_size': 1000,
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite", "device=", "session=", "loss=", "model=", "trainer=", "preprocess=", "override_NL", "pretrained_core=", "defrost", "batch_size="]
    opts, args = getopt.getopt(argv,"n:s:oc:d:l:m:t:p:b:", opt_names)
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
        elif opt in ("--session"):
            session_name = arg
        elif opt in ("--pretrained_core"):
            config["pretrained_core"] = arg
        elif opt in ("--defrost"):
            config["defrost"] = True
        elif opt in ("-b", "--batch_size"):
            config["batch_size"] = int(arg)
            
#%%
# Prepare helpers for training.
config_original = config.copy()
config['device'] = device
print('Device: ', device)
config['seed'] = seed
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
if config['pretrained_core'] is not None:
    pretrained_dir = os.path.join(dirname, 'models', config['pretrained_core'])
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
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device, batch_size=config['batch_size'])

#%%
# Load model and preprocess data.
if from_checkpoint:
    with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
        model = dill.load(f).to(device)
else:
    model = get_model(config)
    if config['pretrained_core'] is not None:
        with open(os.path.join(pretrained_dir, 'model.pkl'), 'rb') as f:
            model_pretrained = dill.load(f).model.core.to(device)
            for i in model_pretrained.parameters():
                i.requires_grad = False
            model.model.core = model_pretrained

if config['defrost']:
    for i in model.model.core.parameters():
        i.requires_grad = True

model.loss, nonlinearity = get_loss(config)
if hasattr(model.loss, 'prepare_loss'):
    model.loss.prepare_loss(train_data, cids=model.cids)
if config['override_output_NL']:
    model.model.output_NL = nonlinearity

if 'binarize' in config['preprocess']: # Binarize the data to eliminate r > 1.
    inds2 = torch.where(train_data['robs'] > 1)[0]
    train_data['robs'][inds2 - 1] = 1
    train_data['robs'][inds2 + 1] = 1
    train_data['robs'][inds2] = 1
    inds2 = torch.where(val_data['robs'] > 1)[0]
    val_data['robs'][inds2 - 1] = 1
    val_data['robs'][inds2 + 1] = 1
    val_data['robs'][inds2] = 1
if 'smooth' in config['preprocess']:
    train_data['robs'] = utils.smooth_robs(train_data['robs'], smoothN=10)
    val_data['robs'] = utils.smooth_robs(val_data['robs'], smoothN=10)
if 'zscore' in config['preprocess']:
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
    dill.dump(config_original, f)