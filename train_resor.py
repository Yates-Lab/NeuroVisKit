'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__
import json, copy
import NDNT
import torch
import dill
from utils.utils import seed_everything, unpickle_data, memory_clear, get_datasets
from utils.train import get_trainer, get_loss
from utils.get_models import get_model
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import utils.postprocess as utils
from sklearn import svm
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm
from unet_resor import build_unet

seed_everything(0)

run_name = 'resorSVC' # Name of log dir.
session_name = '20200304'
nsamples_train=236452
nsamples_val=56643
overwrite = False
from_checkpoint = False
train_device = 'cpu'
device = torch.device(train_device)
seed = 420
config = {
    'loss': 'poisson',
    'model': 'resor',
    'trainer': 'adam',
    'preprocess': 'binarize',
    'bias': True,
}
            
#%%
# Prepare helpers for training.
config_original = config.copy()
config['device'] = device
print('Device: ', device)
config['seed'] = seed
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
tosave_path = os.path.join(checkpoint_dir, 'postprocess')
data_dir = os.path.join(dirname, 'sessions', session_name)
os.makedirs(checkpoint_dir, exist_ok=True)
with open(os.path.join(dirname, 'sessions', session_name, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
print("Input dims:", session['input_dims'])

class RandomConvResBlock(nn.Module):
    def __init__(self, cin, cout, k=3, bias=True, residual=nn.Identity()):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, k, bias=bias, padding=k//2)
        self.residual = 0 if not residual else residual
    def forward(self, x):
        return F.selu(self.conv(x)) + self.residual(x)

def upResidual(cout=None):
    return lambda x: torch.tile(x, (1, math.ceil(cout/x.shape[1]), 1, 1, 1))[:, :cout]
def downResidual(cout=None):
    return lambda x: x[:, :cout]

class RandomResNet(nn.Module):
    def __init__(self, cout, depth=3, cmid=None, k=3, bias=True, seed=420):
        super().__init__()
        seed_everything(seed)
        cmid = cout if cmid is None else cmid
        step_up, step_down, core = [], [], []
        #binary step up from 1 to cmid
        step_up.append(RandomConvResBlock(1, cmid, k, bias, residual=upResidual(cmid)))   
        step_down.append(RandomConvResBlock(cmid, cout, k, bias, residual=downResidual(cout)))
        for i in range(depth - len(step_up) - len(step_down)):
            core.append(
                RandomConvResBlock(cmid, cmid, k, bias)
            )
        self.model = nn.Sequential(
            *[*step_up, *core, *step_down]
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(-1, *session['input_dims'])
        return self.model(x)

# %%
project = False
train = True
if project:
    # Load data.
    train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
    train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device)

    # Train model.
    batch_size = 128
    cout = 1
    depth = 64
    cmid = 64

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # model = RandomResNet(cout, depth=depth, cmid=cmid, k=3, bias=False, seed=420).to(train_device)
    model = build_unet().to(train_device)
    # session['input_dims'] = [1, 32, 32, 16]
    with torch.no_grad():
        train_x_projected = torch.empty((len(train_ds), *session['input_dims']))
        for i, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc='Projecting training data'):
            train_x_projected[i*batch_size:min((i+1)*batch_size, len(train_x_projected))] = model(batch['stim']).detach().cpu()
        torch.save(train_x_projected, 'x_train_projected/0.pt')
        del train_x_projected
        val_x_projected = torch.empty((len(val_ds), *session['input_dims']))
        for i, batch in tqdm(enumerate(val_dl), total=len(val_dl), desc='Projecting validation data'):
            val_x_projected[i*batch_size:min((i+1)*batch_size, len(val_x_projected))] = model(batch['stim']).detach().cpu()
        torch.save(val_x_projected, 'x_val_projected/0.pt')
        del val_x_projected

batch_size = 1000
epochs = 50
val_x_projected = torch.load('x_val_projected/0.pt').to(device)
val_y = torch.load(os.path.join(data_dir, 'val', 'robs.pt')).to(device)[:len(val_x_projected)]
train_x_projected = torch.load('x_train_projected/0.pt').to(device)
train_y = torch.load(os.path.join(data_dir, 'train', 'robs.pt')).to(device)[:len(train_x_projected)]

train_ds = TensorDataset(train_x_projected, train_y)
val_ds = TensorDataset(val_x_projected, val_y)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
session['input_dims'] = [1, 32, 32, 16]
in_size = math.prod([1, 32, 32, 16])

loss_f, _= get_loss({'loss': 'poisson'})

# class Readout(nn.Module):
#     def __init__(self, in_dim, out_dim, loss=NDNT.metrics.poisson_loss.PoissonLoss_datafilter()):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.BatchNorm3d(1),
#             nn.Flatten(),
#             nn.Linear(in_dim, 256),
#             # nn.Tanh(),
#             # nn.Linear(256, 128),
#             nn.Softplus(),
#             nn.Linear(256, out_dim),
#             nn.Softplus()
#         )
#         self.loss = loss
#     def forward(self, x):
#         return self.model(x)
class Readout(nn.Module):
    def __init__(self, in_dim, out_dim, loss=NDNT.metrics.poisson_loss.PoissonLoss_datafilter()):
        super().__init__()
        self.bn = nn.BatchNorm3d(1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(16, 4, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*4, out_dim),
            # nn.Tanh(),
            # nn.Linear(32*32, 16*16),
            # nn.Tanh(),
            # nn.Linear(16*16, out_dim),
            nn.Softplus()
        )
        self.loss = loss
    def forward(self, x):
        x = self.bn(x)
        x = x.reshape(x.shape[0], *x.shape[-3:]).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.bn1(x)
        return self.mlp(x)
    
ml = False
if train:
    if ml:
        model = Readout(in_size, len(session['cids'])).to(device)
        # lbfgs = False
        # if lbfgs:
        #     optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
        #     def closure():
        #         optimizer.zero_grad()
        #         y_pred = model(train_x_projected)
        #         loss = loss_f(y_pred, train_y)
        #         loss.backward()
        #         return loss

        #     best_val_loss = torch.inf
        #     best_model = None
        #     for i in range(epochs):
        #         loss = optimizer.step(closure)
        #         print("epoch: ", i)
        #         print(loss.item(), "lbfsg loss")
        #         with torch.no_grad():
        #             val_loss = loss_f(model(val_x_projected), val_y)
        #         if val_loss < best_val_loss:
        #             best_val_loss = val_loss.item()
        #             del best_model
        #             best_model = copy.deepcopy(model)
        #         print(val_loss.item(), "val_loss")
                
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_val_loss = torch.inf
        best_model = None
        best_model_train = None
        for epoch in tqdm(range(epochs), desc='Epoch'):
            train_loss = 0
            for i, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc='training', leave=False):
                model.zero_grad()
                loss = loss_f(model(batch[0]), batch[1])
                train_loss += loss.item() / len(train_dl)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val_loss = 0
                for i, batch in tqdm(enumerate(val_dl), total=len(val_dl), desc='validation', leave=False):
                    val_loss += loss_f(model(batch[0]), batch[1]) / len(val_dl)
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                del best_model
                best_model = copy.deepcopy(model)
            best_model_train = copy.deepcopy(model)
            print(f'Epoch {epoch} | val_loss: {val_loss.item()} | train_loss: {train_loss}')
        
        with open(os.path.join(checkpoint_dir, 'model.pkl'), 'wb') as f:
            dill.dump(best_model, f)
        with open(os.path.join(checkpoint_dir, 'model_train.pkl'), 'wb') as f:
            dill.dump(best_model_train, f)
    else:
        ids = [30, 36, 22]
        models = []
        for i in ids:
            model = svm.SVC(class_weight='balanced')
            model = model.fit(train_x_projected.reshape(len(train_y), -1), train_y[:, i])
            with torch.no_grad():
                train_loss = 0
                for i, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc='training'):
                    train_loss += loss_f(model.predict(batch[0]), batch[1]) / len(train_dl)
                val_loss = 0
                for i, batch in tqdm(enumerate(val_dl), total=len(val_dl), desc='validation'):
                    val_loss += loss_f(model.predict(batch[0]), batch[1]) / len(val_dl)
                print(f'cid {i} | val_loss: {val_loss.item()} | train_loss: {train_loss.item()}')
            models.append((model, train_loss.item(), val_loss.item()))
        with open(os.path.join(checkpoint_dir, 'models.pkl'), 'wb') as f:
            dill.dump(models, f)
            
    with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
        to_write = {
            'run_name': run_name,
            'session_name': session_name,
            'nsamples_train': nsamples_train,
            'nsamples_val': nsamples_val,
            'seed': seed,
            'config': model.__str__(),
            'device': str(device),
            'best_val_loss': best_val_loss,
        }
        f.write(json.dumps(to_write, indent=2))
else:
    model = dill.load(open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb'))
        
# setattr(model, 'loss', NDNT.metrics.poisson_loss.PoissonLoss_datafilter())
# ev = utils.eval_model_summary(model, val_x_projected)
    
best_cids = [30, 36, 22] #ev.argsort()[::-1]
#%%
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

n = 240*10 # length of movie in frames
win = 240*2
offset = 54137 # index of initial frame
fast = False
movie_cid_list = best_cids[:3] if not fast else []
for cc in movie_cid_list:
    id = session['cids'][cc]
    stims = train_x_projected[offset:offset+n+win].reshape(-1, *session['input_dims']).squeeze()
    robs = train_y[offset:offset+n+win, cc]
    # is_saccade = val_data["fixation_num"][offset:offset+n+win] != val_data["fixation_num"][offset-1:offset-1+n+win]
    probs = model(stims)[:, cc]
    stims = stims[:, 20, :, :]

    stims = stims.detach().cpu().numpy()
    robs = robs.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()

    fig = plt.figure()
    plt.suptitle('Frame 0')
    plt.subplot(3, 1, 1)
    plt.title('Stimulus')
    im1 = plt.imshow(np.zeros((35, win)), cmap='viridis', aspect='auto', animated=True, vmin=-1, vmax=1)
    im1a = plt.gca()
    plt.yticks([])
    plt.xticks([])
    plt.subplot(3, 1, 3)
    plt.title('Response')
    x = np.arange(1, win+1)
    pl1, = plt.plot(x, np.zeros(win), label='true', c='blue')
    pl2, = plt.plot(x, np.zeros(win), label='pred', c='red')
    pla = plt.gca()
    plt.ylim(np.min([robs.min(), probs.min()]), np.max([robs.max(), probs.max()]))
    plt.xlim(0, 23)
    plt.legend(loc='upper right')
    plt.tight_layout()

    def animate(j):
        fig.suptitle(f'Neuron {id}, Frame {j}')
        i = j + win
        im1.set_data(stims[j:i, :, 0].T)
        x = np.arange(j, i)

        pla.set_xlim(j, i)
        pl1.set_data(
            x,
            robs[j:i],
        )
        pl2.set_data(
            x,
            probs[j:i],
        )

        im1a.set_xticklabels([])
        pla.set_xticklabels([])
        return [im1, pl1, pl2]

    fps = 30
    anim = FuncAnimation(
        fig,
        animate,
        frames = n,
        interval = int(n/fps),
        blit=True
    )

    anim.save(tosave_path+'_video_cid%d.mp4'%id, writer = animation.FFMpegWriter(fps=fps))
    del stims, robs, probs
    
pdf_file = PdfPages(tosave_path + '.pdf')
for fig in [plt.figure(n) for n in plt.get_fignums()]:
    fig.savefig(pdf_file, format='pdf')
pdf_file.close()
# memory_clear()
# if from_checkpoint:
#     with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
#         model = dill.load(f).to(device)
# else:
#     model = get_model(config)

# model.loss, nonlinearity = get_loss(config)
# if config['override_output_NL']:
#     model.model.output_NL = nonlinearity

# def smooth_robs(x, smoothN=10):
#     smoothkernel = torch.ones((1, 1, smoothN, 1), device=device) / smoothN
#     out = F.conv2d(
#         F.pad(x, (0, 0, smoothN-1, 0)).unsqueeze(0).unsqueeze(0),
#         smoothkernel).squeeze(0).squeeze(0)  
#     assert len(x) == len(out)
#     return out
# def zscore_robs(x):
#     return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True)

# if config['preprocess'] == 'binarize':
#     inds2 = train_data['robs'] > 1
#     for i in inds2:
#         train_data['robs'][max(i-1, 0):min(i+1, len(train_ds))] = 1
#     inds2 = val_data['robs'] > 1
#     for i in inds2:
#         val_data['robs'][max(i-1, 0):min(i+1, len(val_ds))] = 1
# elif config['preprocess'] == 'smooth':
#     train_data['robs'] = smooth_robs(train_data['robs'], smoothN=10)
#     val_data['robs'] = smooth_robs(val_data['robs'], smoothN=10)
# elif config['preprocess'] == 'zscore':
#     train_data['robs'] = zscore_robs(smooth_robs(train_data['robs'], smoothN=10))
#     val_data['robs'] = zscore_robs(smooth_robs(val_data['robs'], smoothN=10))

# trainer = get_trainer(config)
# best_val_loss = trainer(model, train_dl, val_dl, checkpoint_dir, device)
# #save metadata
# with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
#     to_write = {
#         'run_name': run_name,
#         'session_name': session_name,
#         'nsamples_train': nsamples_train,
#         'nsamples_val': nsamples_val,
#         'seed': seed,
#         'config': config_original,
#         'device': str(device),
#         'best_val_loss': best_val_loss,
#     }
#     f.write(json.dumps(to_write, indent=2))
# %%
