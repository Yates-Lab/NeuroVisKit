'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, memory_clear, get_datasets
from utils.train import get_trainer
from utils.get_models import get_model
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
    'trainer': 'adam',
    'model': 'CNNdense',
    'filters': [20, 20, 20, 20],
    'kernels': [11, 11, 11, 11],
}

if __name__ == "__main__" and "IPython" not in sys.modules:
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite", "device=", "session="]
    opts, args = getopt.getopt(argv,"n:s:oc:d:", opt_names)
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            run_name = arg
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-c", "--config"):
            config = json.loads(arg)
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
    'fix_inds': session['fix_inds'],
})

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
train_dl, val_dl, train_ds, _ = get_datasets(train_data, val_data, device=device)

#%%
# Train model.
memory_clear()
if from_checkpoint:
    with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
        model = dill.load(f).to(device)
else:
    model = get_model(config)
trainer = get_trainer(config)
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
