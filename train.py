'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, memory_clear, get_datasets
from utils.schedule import get_train_loop, get_test_objective
from utils.trainer import train
from utils.train import get_trainer
from utils.get_models import get_model
seed_everything(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Intended device: ', device)

#%%
'''
    User-Defined Parameters
'''
RUN_NAME = 'test' # Name of log dir.
seed = 420
nsamples_train=236452
nsamples_val=56643
overwrite = False
from_checkpoint = False
config = {
    'trainer': 'adam',
    'model': 'CNNdense',
    'filters': [20, 20, 20, 20],
    'kernels': [11, 11, 11, 11],
    'seed': seed,
    'device': device,
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite"]
    opts, args = getopt.getopt(argv,"n:s:oc:", opt_names)
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            RUN_NAME = arg
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

#%%
# Prepare helpers for training.
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, RUN_NAME)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
elif not overwrite and not from_checkpoint:
    print('Directory already exists. Exiting.')
    print('If you want to overwrite, use the -o flag.')
    sys.exit()
    
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
config.update({
    'cids': session['cids'],
    'input_dims': session['input_dims'],
    'mu': session['mu'],
    'fix_inds': session['fix_inds'],
})

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
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
