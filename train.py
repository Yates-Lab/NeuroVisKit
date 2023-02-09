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
from utils.train import TRAINER_DICT, MODEL_DICT
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
config = {
    'trainer': 'adam',
    'model': 'CNNdense',
    'filters': [20, 20, 20, 20],
    'kernels': [11, 11, 11, 11],
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"n:s:oc:",["name=", "seed=", "train=", "val=", "config="])
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            RUN_NAME = arg
        elif opt in ("-s", "--seed"):
            seed_model = int(arg)
        elif opt in ("-c", "--config"):
            config = json.loads(arg)
        elif opt == "--train":
            nsamples_train = int(arg)
        elif opt == "--val":
            nsamples_val = int(arg)
        elif opt in ("-o"):
            overwrite = True

#%%
# Prepare helpers for training.
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, RUN_NAME)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
elif not overwrite:
    print('Directory already exists. Exiting.')
    print('If you want to overwrite, use the -o flag.')
    sys.exit()
    
with open(os.path.join(dirname, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids, input_dims, config['mu'], fix_inds = session['cids'], session['input_dims'], session['mu'], session['fix_inds']

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val)
train_dl, val_dl, train_ds, _ = get_datasets(train_data, val_data, device=device)

#%%
# Train model.
memory_clear()
model = MODEL_DICT(config)
trainer = TRAINER_DICT[config['trainer']]
best_val_loss = trainer(model, train_dl, val_dl, checkpoint_dir, device)