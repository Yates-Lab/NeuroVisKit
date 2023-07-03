
'''
    Preprocess data for training models. (contiguous version)
'''
# Use blocks = true
# treat ds as dataloader (assuming its purely train ds)
# shuffling is done with batch sampler
#%% Imports TODO: make sure shifters are loaded from and stored in the their session folder and not in stim_movies.
import os, sys, getopt, __main__
import numpy as np
import matplotlib.pyplot as plt
import torch
import dill
from copy import deepcopy
from datasets.mitchell.pixel import Pixel, FixationMultiDataset
from datasets.mitchell.pixel.utils import get_stim_list, plot_shifter
from models.utils import plot_stas, eval_model
from models import CNNdense, Shifter
from NDNT.utils import get_max_samples, load_model
from NDNT.training import Trainer, EarlyStopping
from utils.utils import initialize_gaussian_envelope, memory_clear, seed_everything, get_datasets, plot_transients
import utils.preprocess as utils
from matplotlib.backends.backend_pdf import PdfPages
import continuous as C
from utils.utils import memory_clear
from models import Shifter, ModelWrapper
from copy import deepcopy
from NDNT.training import Trainer, EarlyStopping
from NDNT.utils import load_model
from utils.utils import initialize_gaussian_envelope
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from random import shuffle
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
from utils.utils import seed_everything
seed_everything(0)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# '20220216', '20220601', '20220610', '20220805', '20210525', '20191119', '20191120a', '20191121', '20191122', '20191202', '20191205', '20191206', '20191209', '20191210', '20191220', '20191223', '20191226', '20191226', '20191230', '20191231', '20200106', '20200107', '20200109', '20200110', '20200115', '20200226', '20200304'
# to try ['20200304', '20191206', '20191205', '20191122', '20191121', '20191120a', '20191119', '20220610', '20220601']
# for now ['20200304', '20191206', '20191205', '20191121', '20191120a'] #these are kilo '20220610', '20220601'
# didnt work: '20191231', '20191122', '20191119', 

session_name = '20200304C'
datadir = [
    '/Data/stim_movies/',
    '/mnt/Data/Datasets/MitchellV1FreeViewing/stim_movies/'
    ][0]
spike_sorting = 'kilo'
num_lags = 36

if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"t:s:l:p:d:",["train_shifter=", "session=", "lags=", "path=", "device=", "sorting="])
    for opt, arg in opts:
        if opt in ("-t", "--train_shifter"):
            train_shifter = arg.upper() == 'TRUE'
        elif opt in ("-s", "--session"):
            session_name = arg
        elif opt in ("-l", "--lags"):
            num_lags = int(arg)
        elif opt in ("-p", "--path"):
            datadir = arg
        elif opt in ("-d", "--device"):
            device = torch.device(arg)
        elif opt in ("--sorting"):
            spike_sorting = arg

WINDOW_SIZE = None#35
TRAIN_FRAC = 0.85
APPLY_SHIFTER = True
BATCH_SIZE = 1000
VALID_EYE_RAD = 5.2
SEED = 1234
#%%
val_device = device # if you're cutting it close, put the validation set on the cpu
sesslist = list(get_stim_list().keys())
assert session_name in sesslist, "session name %s is not an available session" %session_name
NBname = f'shifter_{session_name}_{SEED}'
outdir = os.path.join(os.getcwd(), 'data', 'sessions', session_name)
if not os.path.exists(outdir):
    print('Creating directory: ', outdir)
    os.makedirs(outdir)
else:
    print("Saving to:", outdir)
cp_dir = os.path.join(outdir, NBname)
FRAC_DF_INCLUDE=0.2

def get_fix_dataloader(ds, inds, batch_size=1, num_workers=os.cpu_count()//2):
    sampler = BatchSampler(
        SubsetRandomSampler(inds),
        batch_size=batch_size,
        drop_last=True
    )
    dl = DataLoader(ds, sampler=sampler, batch_size=None, num_workers=num_workers)
    return dl

ds = FixationMultiDataset(sess_list=session_name.split('_'),
    dirname=datadir,
    stimset='Train',
    binarize_spikes=False,
    requested_stims=['Gabor', 'Dots', 'Grating', 'BackImage'],
    downsample_s=1,
    downsample_t=1,
    num_lags=num_lags,
    num_lags_pre_sac=40,
    spike_sorting=spike_sorting,
    saccade_basis = None,
    download=True,
    flatten=False,
    crop_inds=[5, 75, 5, 75],
    min_fix_length=50,
    max_fix_length=1000,
    valid_eye_rad=VALID_EYE_RAD,
    add_noise=0,
    use_blocks=True,
    verbose=True,
    
)

print('%d valid blocks' %len(ds))
block = 0
ifix = 0
inds = ds.get_stim_indices('Gabor')
batch = ds[inds[0]]
plt.imshow(batch['stim'][0,0,:,:].squeeze().cpu().numpy(), interpolation='none', aspect='auto', cmap='gray')

ds.use_blocks = False
stim_id = 0
lag = 5
sta = C.check_raw_sta(ds, lag=lag, stim_id=stim_id)
ds.use_blocks = True
stas = ds.get_stas()
mu, bestlag, _ = plot_stas(stas.detach().cpu().numpy())
#%% check batches
nspikes, sus_fix = C.check_spikes_per_batch(ds)

#%% make dataloaders
seed_everything(SEED)
batch_size = 3
ds.use_blocks = True
inds = list(range(len(ds)))
shuffle(inds)
train_inds = inds[:int(TRAIN_FRAC*len(inds))]
val_inds = inds[int(TRAIN_FRAC*len(inds)):]

train_dl = get_fix_dataloader(ds, train_inds, batch_size=batch_size)
val_dl = get_fix_dataloader(ds, val_inds, batch_size=batch_size)

rsum = 0
n = 0
for batch in tqdm(train_dl):
    rsum += batch['robs'].sum(dim=0)
    n += batch['robs'].shape[0]

rbar = rsum/n

cids = np.where(rsum > 1000)[0]
rsum = rsum[cids]

# # Analyze dataset.
# stas, cids, gab_inds = utils.get_ds_analysis(ds, WINDOW_SIZE, FRAC_DF_INCLUDE)
# mu, bestlag, _ = plot_stas(stas.detach().numpy())
# f = utils.plot_sacta(ds)

# #%% Data storage.
# train_data, val_data, train_inds, val_inds = utils.get_data_from_ds(ds)

# print('New stim shape {}'.format(train_data['stim'].shape))
# indsG = np.where(np.in1d(val_inds, gab_inds))[0]
# indsN = np.where(~np.in1d(val_inds, gab_inds))[0]

# print('{} training samples, {} validation samples, {} Gabor samples, {} Image samples'.format(len(train_inds), len(val_inds), len(indsG), len(indsN)))

# cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FRAC_DF_INCLUDE)[0]
# cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

# Store data and get dict of paths.
with open(os.path.join(outdir, 'metadata.txt'), 'w') as f:
    f.write(f'Session: {session_name};\n\
            num_lags: {num_lags};\n\
            apply_shifter: {APPLY_SHIFTER};\n\
            window_size: {WINDOW_SIZE};\n\
            frac_df_include: {FRAC_DF_INCLUDE};\n\
            num_neurons: {len(cids)};\n\
            num_train: {len(train_inds)};\n\
            num_val: {len(val_inds)};\n\
            train_frac: {TRAIN_FRAC};\n')
paths = ds.save(os.path.join(outdir, 'ds.pkl'))

#%% Store session info.
session = {
    'fix_inds_train': None,
    'fix_inds_val': None,
    'cids': cids,
    'mu': mu.copy(),
    'input_dims': ds.dims + [num_lags],
    'train_inds': train_inds,
    'val_inds': val_inds,
    'num_lags': num_lags,
}
with open(os.path.join(outdir, 'session.pkl'), 'wb') as f:
    dill.dump(session, f)
#%%


# inds = ds.get_stim_indices("Gabor")
# robs = ds[inds]["robs"]
# n, nrobs = len(robs), robs.shape[1]
# def correlate(i, j, c):
#     a, b = robs[:n-c, i], robs[c:, j]
#     return a.dot(b) / (a.norm(2) * b.norm(2))
# cc = torch.zeros(nrobs, nrobs)
# for c in range(10):
#     for i in range(nrobs):
#         for j in range(nrobs):
#             cc[i, j] += correlate(i, j, c)
#     plt.figure()
#     plt.imshow(cc)
#     plt.title(f'{c}')

# nbatches = 10
# dl = get_fix_dataloader(ds, inds)
# plt.figure(figsize=(5*2, 2*nbatches))
# for batchI, batch in tqdm(enumerate(dl)):
#     if batchI >= nbatches:
#         break
#     robs = batch["robs"]
#     n, nrobs = len(robs), robs.shape[1]
#     def correlate(i, j, c):
#         a, b = robs[:n-c, i], robs[c:, j]
#         norm = (a.norm(2) * b.norm(2))
#         return a.dot(b) / norm if norm else 0
#     cc = torch.zeros(nrobs, nrobs)
#     for c in range(5):
#         for i in range(nrobs):
#             for j in range(nrobs):
#                 cc[i, j] += correlate(i, j, c)
#                 assert np.isfinite(cc[i, j])
#         plt.subplot(nbatches, 5, batchI*5 + c+1)
#         plt.imshow(cc)
#         plt.axis("off")
#         plt.title(f'{c}, fixation {batchI}')
# plt.tight_layout()
# plt.savefig("cov.png")