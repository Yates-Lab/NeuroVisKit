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
import torch.nn.functional as F
import torch.nn as nn
from NDNT.training.trainer import Trainer, LBFGSTrainer
from NDNT.training.lbfgsnew import LBFGSNew
from NDNT.training.lbfgs import LBFGS
import NDNT.modules.layers as layers
from NDNT.metrics.poisson_loss import PoissonLoss_datafilter
from NDNT.training.earlystopping import EarlyStopping
from models.base import ModelWrapper
from utils.get_models import PytorchWrapper
from utils.postprocess import eval_model_summary

import numpy as np
seed_everything(0)

class Dense(nn.Module):
    def __init__(self, input_dims, outc, weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_):
        super().__init__()
        self.input_dims = input_dims
        self.weights = nn.Parameter(
            weight_init(torch.empty((1, *input_dims, outc)))
        )
        self.bias = nn.Parameter(
            bias_init(torch.empty(1, outc))
        )
    def forward(self, x):
        x = x.reshape(x.shape[0], *self.input_dims)
        return x @ self.weights + self.bias

class torchGLM(nn.Module):
    def __init__(self, input_dims, outc, output_NL=nn.Softplus()):
        super().__init__()
        self.core = nn.Linear(np.prod(input_dims), outc)
        self.output_NL = output_NL
    def forward(self, x):
        return self.output_NL(self.core(x.reshape(x.shape[0], -1)))
    
def gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    exp = torch.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = torch.sin((x-cx)*fx+(y-cy)*fy+(t-ct)*ft+p)
    return exp*sin
    
class Gabor(nn.Module):
    def __init__(self, input_dims, NC):
        super().__init__()
        self.params = nn.Parameter(torch.empty(11, NC, 1, 1, 1).uniform_(0.1, 0.9))
        self.input_dims = input_dims[1:]
        self.NC = NC
    def forward(self, x):
        x = x.reshape(x.shape[0], *self.input_dims)
        grids = torch.meshgrid(*[torch.linspace(0, 1, i) for i in self.input_dims])
        grids = [i.unsqueeze(0) for i in grids]
        return torch.einsum('nxyt,bxyt->bn', gabor_timedomain(*grids, *self.params), x)

class GaborGLM(nn.Module):
    def __init__(self, input_dims, NC, output_NL=nn.Softplus()):
        super().__init__()
        self.core = nn.Sequential(Gabor(input_dims, NC))
        self.output_NL = output_NL
    def forward(self, x):
        return self.output_NL(self.core(x))
        
class GLM(torch.nn.Module):
    
    def __init__(self, 
                 intput_dims,
                 NC,
                 cids=None,
                 reg_vals = None):
        seed_everything(seed)
        super().__init__()
        
        self.dims = intput_dims
        self.cids = cids
        self.loss = PoissonLoss_datafilter()
        
        # build model
        self.layer = layers.NDNLayer(input_dims=self.dims,
                              num_filters=NC, reg_vals=reg_vals, bias=True,
                              NLtype='softplus')
        self.core = nn.Sequential(
            self.layer,
        )
        
    def forward(self, input):
        
        y = self.layer(input['stim'])
        
        return y
    
    def prepare_regularization(self, normalize_reg=False):
        
        self.layer.reg.normalize = normalize_reg
        self.layer.reg.build_reg_modules()
        
    def compute_reg_loss(self):
        
        return self.layer.compute_reg_loss()
    
    # def training_step(self, batch):
        
    #     y = batch['robs']
    #     dfs = batch['dfs']

    #     y_hat = self(batch)
        
    #     loss = self.loss(y_hat, y[..., self.cids], dfs[..., self.cids])

    #     regularizers = self.compute_reg_loss()

    #     return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}
    
    # def validation_step(self, batch):
        
    #     y = batch['robs']
    #     dfs = batch['dfs']

    #     y_hat = self(batch)

    #     loss = self.loss(y_hat, y[..., self.cids], dfs[..., self.cids])
        
    #     return {'loss': loss, 'val_loss': loss, 'reg_loss': None}

run_name = 'test_glm_gabor' # Name of log dir.
session_name = '20200304'
nsamples_train=None
nsamples_val=56643
overwrite = True
from_checkpoint = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 420
config = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'gabor_cnn', # utils/get_models.py for more model options
    'trainer': 'Adam', # utils/train.py for more trainer options
    'preprocess': [],# this further preprocesses the data before training
    'override_output_NL': False, # this overrides the output nonlinearity of the model according to the loss,
    'pretained_core': None, # this is the name of a pretrained core to load
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite", "device=", "session=", "loss=", "model=", "preprocess=", "override_NL", "pretrained_core=", "defrost"]
    opts, args = getopt.getopt(argv,"n:s:oc:d:l:m:p:", opt_names)
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
            
#%%
# Prepare helpers for training.
config_original = config.copy()
config['device'] = device
print('Device: ', device)
config['seed'] = seed
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
if 'pretrained_core' in config and config['pretrained_core'] is not None:
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
input_dims = session['input_dims']
config.update({
    'cids': session['cids'],
    'input_dims': input_dims,
    'mu': session['mu'],
})

# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)

print('Preprocessing data...')
if 'gabor' in config['preprocess']:
    inds = np.where(train_data['stimid'] == 0)[0]
    for k in train_data.keys():
        train_data[k] = train_data[k][inds]
    print(f'Using only Gabor stimuli. {len(train_data["stim"])} samples.')
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
if 'rfft' in config['preprocess']:
    train_data['stim'] = torch.fft.rfftn(train_data['stim'].reshape(-1, *config['input_dims']), dim=len(config['input_dims']), mode='forward')
    val_data['stim'] = torch.fft.rfftn(val_data['stim'].reshape(-1, *config['input_dims']), dim=len(config['input_dims']), mode='forward')
    input_dims = val_data['stim'].shape[1:]

train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=torch.device('cpu'))

# #%%
# from utils.postprocess import eval_model_fast
# from itertools import product
# from tqdm import tqdm
# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout
# hp_ranges = [1e-3, 1e-2, 1e-1, 1, 10, 100]
# hp_combos = list(product(hp_ranges, repeat=3))
# cells = range(len(session['cids']))
# scores = [np.inf for _ in cells]
# hps = {
#     'd2x': [None]*len(cells),
#     'd2t': [None]*len(cells),
#     'glocalx': [None]*len(cells),
# }
# errors, untried_pairs = [], []
# for cell in tqdm(cells):
#     for d2x, d2t, glocalx in tqdm(hp_combos, leave=False):
#         try:
#             with HiddenPrints():
#                 model = GLM(
#                     input_dims,
#                     len(config['cids']),
#                     #0.01, 0.11, 10
#                     reg_vals={'d2x':d2x, 'd2t':d2t, 'glocalx':glocalx},
#                     cids=config['cids'],
#                 )
#                 model = ModelWrapper(model)
#                 model.prepare_regularization()
#                 model = model.to(dtype=train_data['stim'].dtype)
#                 seed_everything(seed)
#                 optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
#                 trainer = LBFGSTrainer(
#                     optimizer=optimizer,
#                     device=device,
#                     optimize_graph=True,
#                     dirpath=False,
#                     verbose=2,
#                 )
#                 print('Training...')
#                 fitted = trainer.fit(model, train_data, val_data)
#                 score = eval_model_fast(model, val_data)[cell]
#                 if score > scores[cell]:
#                     scores[cell] = score
#                     hps['d2x'][cell] = d2x
#                     hps['d2t'][cell] = d2t
#                     hps['glocalx'][cell] = glocalx
#                 del model, fitted, trainer
#         except Exception as e:
#             errors.append(str(e))
#             untried_pairs.append((cell, session['cids'][cell], d2x, d2t, glocalx))
# with open(os.path.join(checkpoint_dir, 'hp.txt'), 'w') as f:
#     to_write = {
#         'scores': str(scores),
#         'd2x': str(hps['d2x']),
#         'd2t': str(hps['d2t']),
#         'glocalx': str(hps['glocalx']),
#         'hp_ranges': str(hp_ranges),
#         'errors': str(errors),
#         'untried_pairs': str(untried_pairs),
#     }
#     f.write(json.dumps(to_write, indent=2))
# quit()
#%%
from models.utils import plot_sta_movie
from utils.get_models import get_model
def gabor_timedomain(x, y, t, fx, fy, ft, cx, cy, ct, sx, sy, st, p):
    exp = torch.exp(-0.5*((x-cx)/sx)**2-0.5*((y-cy)/sy)**2-0.5*((t-ct)/st)**2)
    sin = torch.sin((x-cx)*fx+(y-cy)*fy+(t-ct)*ft+p)
    return exp*sin
    
class Gabor(nn.Module):
    def __init__(self, input_dims, NC, max_batch_size=1000, frozen_center=False, bias=True):
        super().__init__()
        self.input_dims = input_dims[1:]
        self.NC = NC
        self.max_batch_size = max_batch_size
        self.gain = nn.Parameter(torch.ones(1, NC))
        self.bias = nn.Parameter(torch.zeros(1, NC)) if bias else 0
        #init parameters
        inits = torch.tensor([1, 1, 1, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0]).reshape(10, 1, 1, 1, 1)
        self.fparams = nn.Parameter(torch.ones(3, NC, 1, 1, 1)*inits[:3] + torch.randn(3, NC, 1, 1, 1)*0.01)
        self.spparams = nn.Parameter(torch.ones(4, NC, 1, 1, 1)*inits[6:10] + torch.randn(4, NC, 1, 1, 1)*0.01)
        cparams = torch.ones(3, NC, 1, 1, 1)*inits[3:6]+torch.randn(3, NC, 1, 1, 1)*0.01*int(not frozen_center)
        self.cparams = nn.Parameter(cparams, requires_grad=not frozen_center)
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, *self.input_dims)
        gabors = self.get_kernels()
        if len(x) > self.max_batch_size:
            return torch.cat([torch.sum(x[i:i+self.max_batch_size]*gabors, dim=(2, 3, 4)) for i in range(0, len(x), self.max_batch_size)])
        return torch.sum(x*gabors, dim=(2, 3, 4)) * self.gain + self.bias
    def get_kernels(self):
        gridx = torch.linspace(0, 1, self.input_dims[0], device=self.fparams.device).reshape(1, -1, 1, 1)
        gridy = torch.linspace(0, 1, self.input_dims[1], device=self.fparams.device).reshape(1, 1, -1, 1)
        gridt = torch.linspace(0, 1, self.input_dims[2], device=self.fparams.device).reshape(1, 1, 1, -1)
        return gabor_timedomain(gridx, gridy, gridt, *self.fparams, *self.cparams, *self.spparams)

class GaborGLM(nn.Module):
    def __init__(self, input_dims, NC, output_NL=nn.Softplus()):
        super().__init__()
        self.core = nn.Sequential(Gabor(input_dims, NC))
        self.output_NL = output_NL
    def forward(self, x):
        return self.output_NL(self.core(x))
    def get_filters(self):
        return self.core[0].get_kernels().detach().cpu()
    
class GaborBasisGLM(nn.Module):
    def __init__(self, input_dims, basis_dim, NC, output_NL=nn.Softplus()):
        super().__init__()
        self.core = nn.Sequential(GaborGLM(input_dims, basis_dim, output_NL=nn.Identity()))
        self.readout = nn.Linear(basis_dim, NC)
        self.output_NL = output_NL
    def forward(self, x):
        return self.output_NL(self.readout(self.core(x)))
    
class GaborConvLayer(nn.Module):
    # expects to be shaped (batch, cin, x, y, t)
    def __init__(self, cin, cout, k=3, **kwargs):
        super().__init__()
        k = (k, k, k) if isinstance(k, int) else k
        self.gabor = Gabor((1, *k), cin*cout, frozen_center=True)
        self.cout = cout
        self.cin = cin
        self.kwargs = kwargs
    def forward(self, x):
        kernels = self.gabor.get_kernels()
        return F.conv3d(x, kernels.reshape(self.cout, self.cin, *kernels.shape[1:]), **self.kwargs)
    
class GaborConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, NL=nn.Softplus(), **kwargs):
        super().__init__()
        self.conv = GaborConvLayer(cin, cout, k=k, **kwargs)
        self.NL = NL
        self.bn = nn.BatchNorm3d(cin)
    def forward(self, x):
        return self.NL(self.conv(self.bn(x)))

class FoldTimetoChannel(nn.Module):
    def forward(self, x):
        return x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], x.shape[1]*x.shape[4], x.shape[2], x.shape[3])
class Reshape(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims
    def forward(self, x):
        return x.reshape(x.shape[0], *self.input_dims)
class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
    
class GaborCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(
            Reshape(config["input_dims"]),
            nn.Conv3d(1, 3, kernel_size=3, padding="same"),
            GaborConvBlock(3, 20, k=11, NL=nn.Tanh(), padding="same"),
            FoldTimetoChannel(),
            nn.GroupNorm(20, 20*24),
            nn.Conv2d(20*24, 20, kernel_size=1),
            nn.Softplus(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 20, kernel_size=11, padding=2, stride=2),
            nn.Softplus(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(980, 62),
            nn.Softplus(),
        )
    def forward(self, x):
        return self.core(x)
    def get_gabor_filters(self):
        return self.core[2].conv.gabor.get_kernels().detach().cpu()
    def plot_gabor_filters(self):
        plot_sta_movie(self.get_gabor_filters().permute(3, 1, 2, 0).numpy(), is_weights=False)
        
    # dense_cnn = get_model({
    #     'model': 'CNNdense', # utils/get_models.py for more model options
    #     'device': device,
    #     'filters': [20], # the config is fed into the model constructor
    #     'kernels': [11],
    #     'seed': seed,
    #     'input_dims': input_dims,
    #     'cids': config["cids"],
    #     **session
    # })
    
    
print('Loading model...')
if from_checkpoint:
    print ("Loading from checkpoint")
    with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
        model = dill.load(f)
else:
    seed_everything(seed)
    if config['model'] == "GLM":
        model = ModelWrapper(GLM(
            input_dims,
            len(config['cids']),
            #0.01, 0.11, 10
            reg_vals={'d2x':0.01, 'd2t':0.11, 'glocalx':10},
            cids=config['cids'],
        ))
        model.prepare_regularization()
    elif config['model'] == "torchGLM":
        model = PytorchWrapper(torchGLM(
            input_dims=input_dims,
            outc=len(config['cids']),
        ), cids=config['cids'])
    elif config['model'] == "gabor":
        model = PytorchWrapper(GaborGLM(
            input_dims,
            len(config['cids']),
        ), cids=config['cids'])
    elif config['model'] == "gabor_basis":
        model = PytorchWrapper(GaborBasisGLM(
            input_dims,
            100,
            len(config['cids']),
        ), cids=config['cids'])
    elif config['model'] == "gabor_cnn":
        model = PytorchWrapper(GaborCNN(), cids=config['cids'])
    else:
        raise ValueError("Model not recognized; only GLMs are supported.")
model = model.to(device=device, dtype=train_data['stim'].dtype)

print('Training...')
seed_everything(seed)
if config['trainer'] == "LBFGS":
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
    trainer = LBFGSTrainer(
        optimizer=optimizer,
        device=device,
        optimize_graph=True,
        dirpath=False,
        verbose=2,
        # max_epochs=10,
        # early_stopping=EarlyStopping(patience=30, verbose=True, delta=1e-6),
    )
    fitted = trainer.fit(model, train_data, val_data)
elif config['trainer'] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        optimizer=optimizer,
        device=device,
        optimize_graph=False,
        dirpath=False,
        verbose=2,
        early_stopping=EarlyStopping(patience=30, verbose=True, delta=1e-6),
    )
    fitted = trainer.fit(model, train_dl, val_dl)
plot_sta_movie(model.model.get_filters().permute(3, 1, 2, 0).numpy(), is_weights=False)
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
        # 'best_val_loss': best_val_loss,
    }
    f.write(json.dumps(to_write, indent=2))
with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as f:
    dill.dump(config_original, f)
with open(os.path.join(checkpoint_dir, 'model.pkl'), 'wb') as f:
    dill.dump(model, f)
# %%
