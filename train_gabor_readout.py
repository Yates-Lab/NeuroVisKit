'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__
import json
import torch
import dill
import numpy as np
from utils.utils import seed_everything, unpickle_data, get_datasets
import utils.train as utils
from utils.loss import get_loss
from utils.get_models import get_model, PytorchWrapper
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
seed_everything(0)

run_name = 'test_gabor_readout' # Name of log dir.
session_name = '20200304'
nsamples_train=None
nsamples_val=None
overwrite = True
from_checkpoint = False
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
seed = 420
config = {
    'loss': 'bits_per_spike', # utils/loss.py for more loss options
    'model': 'CNNdense', # utils/get_models.py for more model options
    'trainer': 'adam', # utils/train.py for more trainer options
    'filters': [20, 20, 20, 20], # the config is fed into the model constructor
    'kernels': [11, 11, 11, 11],
    'preprocess': [],# this further preprocesses the data before training
    'override_output_NL': False, # this overrides the output nonlinearity of the model according to the loss,
    'pretrained_core': None,
    'defrost': False,
    'batch_size': 256,
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
def random_onehot(nrows, ncols):
    x = torch.ones(nrows, ncols)/1000
    x[torch.arange(nrows), torch.randperm(ncols)[:nrows]] = 1
    return x

def hideTimeInBatch(x):
    # x shape is b, c, x, y, t
    # returns b*t, c, x, y
    return x.permute(0, 4, 1, 2, 3).reshape(x.shape[0]*x.shape[4], x.shape[1], x.shape[2], x.shape[3])
def unhideTimeInBatch(x, nt):
    # x shape is b*t, c, x, y
    # returns b, c, x, y, t
    return x.reshape(int(x.shape[0]/nt), nt, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 3, 4, 1)
def oddify(i):
    return i + i % 2 - 1
def locality(x):
    # calculates the locality of the energy of a batch of images in shape b, *dims
    axes = torch.meshgrid(*[torch.linspace(-1, 1, oddify(i)) for i in x.shape[1:]], indexing='xy')
    locality_kernel = 0
    for ax in axes:
        locality_kernel = ax**2 + locality_kernel
    locality_kernel = locality_kernel.to(x.device).unsqueeze(0).unsqueeze(0)
    if len(axes) == 1:
        return F.conv1d(x.unsqueeze(1)**2, locality_kernel, padding="valid").mean()
    elif len(axes) == 2:
        return F.conv2d(x.unsqueeze(1)**2, locality_kernel, padding="valid").mean()
class GaborReadout2D(nn.Module):
    def __init__(self, nx, ny, nt, cids):
        super().__init__()
        self.cids = cids
        self.dims = (nx, ny, nt)
        self.filters = self.get_kernels(np.load('gabor_params_2d.npy'))[:, None, ...]
        self.filters = nn.Parameter(torch.from_numpy(self.filters), requires_grad=False)
        self.means = nn.Parameter(torch.zeros(1, len(self.filters), 1, 1, 1), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(1, len(self.filters), 1, 1, 1), requires_grad=False)
        self.svectors = nn.Parameter(torch.ones(len(cids), nx, ny)/1000)
        self.conv = nn.Sequential(
            nn.Conv3d(len(self.filters)*2, len(self.filters), (1, 1, 11), padding="same"),
            nn.Conv3d(len(self.filters), 20, 3, padding="same")
        )
        self.tvectors = nn.Parameter(torch.ones(len(cids), nt)/1000)
        self.cvectors = nn.Parameter(torch.randn(len(cids), 20))
        self.output_NL = nn.Softplus()
        self.penalties_mean = [0]*6
        self.penalties_deviations = [1]*6
    def compute_reg_loss(self, *args, **kwargs):
        penalties = [
            self.tvectors.norm(1),
            self.svectors.norm(1),
            self.cvectors.norm(1),
            self.cvectors.norm(2)/10,
            locality(self.svectors),
            locality(self.tvectors),
        ]
        self.penalties_mean = [0.9*self.penalties_mean[i] + 0.1*penalties[i] for i in range(len(penalties))]
        deviations = [(self.penalties_mean[i] - penalties[i])**2 for i in range(len(penalties))]
        self.penalties_deviations = [0.9*self.penalties_deviations[i] + 0.1*deviations[i] for i in range(len(penalties))]
        return sum(penalties) / len(penalties)
    def get_kernels(self, params):
        x, y = np.meshgrid(*[np.linspace(0, 1, i+1-i%2) for i in self.dims[:2]])
        x, y = [i.reshape(1, *i.shape) for i in [x, y]]
        return self.gabor(x, y, *params.T[..., None, None]).astype(np.float32)
    def gabor(self, x, y, fx, fy, sx, sy, p):
        cx, cy = 0.5, 0.5
        exp = np.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2)
        sin = np.sin(2*np.pi*(x-cx)*fx+2*np.pi*(y-cy)*fy+p)
        return exp*sin
    def prepare_model(self, training_ds, overwrite=False):
        with torch.no_grad():
            if os.path.exists("means2D.pt") and not overwrite:
                self.means = torch.load("means2D.pt").to(device)
            else:
                running_sum, numel = torch.zeros(self.means.numel(), device=self.means.device), 0
                for batch in tqdm(DataLoader(training_ds, batch_size=1000)):
                    x = hideTimeInBatch(batch["stim"].reshape(-1, 1, *self.dims))
                    numel += x.numel()
                    running_sum += F.conv2d(x, self.filters, padding="same").mean((0, 2, 3))
                self.means[0, :, 0, 0, 0] = running_sum/numel
                torch.save(self.means, "means2D.pt")
            if os.path.exists("stds2D.pt") and not overwrite:
                self.stds = torch.load("stds2D.pt").to(device)
            else:
                running_sum, numel = torch.zeros(self.means.numel(), device=self.means.device), 0
                for batch in tqdm(DataLoader(training_ds, batch_size=500)):
                    x = hideTimeInBatch(batch["stim"].reshape(-1, 1, *self.dims))
                    numel += x.numel()
                    running_sum += (F.conv2d(x, self.filters, padding="same")-self.means[..., 0]).pow(2).sum((0, 2, 3))
                self.stds[0, :, 0, 0, 0] = (running_sum/numel)**0.5                
                torch.save(self.stds, "stds2D.pt")
    def transform_data(self, x):
        x = hideTimeInBatch(x.reshape(x.shape[0], 1, *self.dims))
        filtered = F.conv2d(x, self.filters, padding="same")
        filtered = unhideTimeInBatch(filtered, self.dims[2])
        filtered = (filtered - self.means)/self.stds + self.means
        return filtered #torch.einsum('bcxyt,oct->boxy', filtered, self.channel_bottleneck)
    def forward(self, x):
        filtered = self.transform_data(x) if x.shape[1] != len(self.filters) else x
        filtered = torch.concat([F.relu(filtered), F.relu(-filtered)], dim=1)
        filtered = self.conv(filtered) # output shaped b, 20, x, y, t
        filtered = torch.einsum('bcxyt,nc->bnxyt', filtered, self.cvectors)
        filtered = torch.einsum('bnxyt,nxy,nt->bn', filtered, self.svectors, self.tvectors)
        return self.output_NL(filtered)
    
class GaborReadout(nn.Module):
    def __init__(self, nx, ny, nt, cids):
        super().__init__()
        self.cids = cids
        self.dims = (nx, ny, nt)
        self.filters = self.get_kernels(np.load('gabor_params.npy'))[:, None, ...]
        self.filters = nn.Parameter(torch.from_numpy(self.filters), requires_grad=False)
        self.means = nn.Parameter(torch.zeros(1, len(self.filters), 1, 1, 1), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(1, len(self.filters), 1, 1, 1), requires_grad=False)
        # self.channel_bottleneck = nn.Parameter(random_onehot(len(cids), len(self.filters)))
        self.channel_bottleneck = nn.Parameter(torch.randn(nt, len(self.filters), nt), requires_grad=False)
        self.tvectors = nn.Parameter(torch.ones(len(cids), nt)/1000)
        self.svectors = nn.Parameter(torch.ones(len(cids), nx, ny)/1000)
        self.cvectors = nn.Parameter(torch.randn(len(cids), len(cids)*2))
        self.output_NL = nn.Softplus()
    def compute_reg_loss(self, *args, **kwargs):
        return self.tvectors.norm(1) + self.svectors.norm(1) + self.cvectors.norm(1)
    def get_kernels(self, params):
        x, y, t = np.meshgrid(*[np.linspace(0, 1, i+1-i%2) for i in self.dims])
        x, y, t = [i.reshape(1, *i.shape) for i in [x, y, t]]
        return self.gabor(x, y, t, *params.T[..., None, None, None]).astype(np.float32)
    def gabor(self, x, y, t, fx, fy, ft, sx, sy, st, p):
        cx, cy, ct = 0.5, 0.5, 0.5
        exp = np.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
        sin = np.sin(2*np.pi*(x-cx)*fx+2*np.pi*(y-cy)*fy+2*np.pi*(t-ct)*ft+p)
        return exp*sin
    def prepare_model(self, training_ds):
        if os.path.exists("means.pt") and os.path.exists("stds.pt") and os.path.exists("gabords.pt"):
            self.means = torch.load("means.pt").to(device)
            self.stds = torch.load("stds.pt").to(device)
            return
        dl = DataLoader(training_ds, batch_size=100)
        with torch.no_grad():
            # for i, filt in tqdm(enumerate(self.filters), desc="Computing means and stds"):
            #     running_sum, numel = 0, 0
            #     for batch in tqdm(dl):
            #         x = batch["stim"].reshape(-1, 1, *self.dims)
            #         numel += x.numel()
            #         running_sum += F.conv3d(x, filt.unsqueeze(0), padding="same").mean()
            #     self.means[:, i] = running_sum/numel
            #     running_sum, numel = 0, 0
            #     for batch in tqdm(dl):
            #         x = batch["stim"].reshape(-1, 1, *self.dims)
            #         numel += x.numel()
            #         running_sum += (F.conv3d(x, filt.unsqueeze(0), padding="same")-self.means[:, i]).pow(2).sum()
            #     self.stds[:, i] = (running_sum/numel)**0.5
            # torch.save(self.means, "means.pt")
            # torch.save(self.stds, "stds.pt")
            gabords = torch.empty(len(train_ds), self.dims[-1], *self.dims[:-1])
            i = 0
            for batch in tqdm(dl):
                gabords[i:i+batch["stim"].shape[0]] = self.transform_data(batch["stim"]).detach().cpu()
                i += batch["stim"].shape[0]
            torch.save(gabords, "gabords.pt")
    def transform_data(self, x):
        x = x.reshape(x.shape[0], 1, *self.dims)
        filtered = F.conv3d(x, self.filters, padding="same")
        filtered = (filtered - self.means)/self.stds + self.means
        return torch.einsum('bcxyt,oct->boxy', filtered, self.channel_bottleneck)
    def forward(self, x):
        filtered = self.transform_data(x) if x.shape[1] != len(self.filters) else x
        filtered = torch.einsum('bcxyt,nxy,nt->bcn', filtered, self.svectors, self.tvectors)
        filtered = torch.concat([F.relu(filtered), F.relu(-filtered)], dim=1)
        filtered = torch.einsum('bcn,nc->bn', filtered, self.cvectors)
        return self.output_NL(filtered)
model = PytorchWrapper(GaborReadout2D(*config['input_dims'][-3:], config['cids']), config['cids']).to(device)
if hasattr(model.model, 'prepare_loss'):
    model.model.prepare_loss(train_ds)
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