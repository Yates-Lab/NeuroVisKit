'''
    Analysis of trained models.
'''
#%%
#!%load_ext autoreload
#!%autoreload 2
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import NDNT
import numpy as np
import dill
import os
from utils.utils import plot_transientsC, TimeLogger, to_device
from _utils.utils import isInteractive, get_opt_dict, seed_everything, joinCWD
import utils.postprocess as utils
import utils.lightning as lutils
from models.utils.plotting import plot_sta_movie
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from datasets.mitchell.pixel.fixation import FixationMultiDataset
from matplotlib.animation import FuncAnimation
from utils.mei import irfC
import matplotlib.animation as animation
#%%
seed_everything(0)

config = {
    'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
    'session': '20200304C',
    'name': 'cnnc',
    # 'pytorch': False,
    'fast': False,
    # 'load_preprocessed': False,
}

# Here we make sure that the script can be run from the command line.
if not isInteractive():
    loaded_opts = get_opt_dict([
        ('n:', 'name='),
        ('d:', 'device=', torch.device),
        ('s:', 'session='),
        # ('p', 'pytorch'),
        ('f', 'fast'),
        # ('l', 'load_preprocessed'),
    ], default=config)
else:
    config['name'] = 'test_readout_nowindow'
    # pass # here you can interactively overwrite config options config["option"] = <new value>
#%%
# Determine paths.
device = config['device']
print("using device", str(device))
config["dirname"] = joinCWD('data')
dirs = lutils.get_dirnames(config)
tosave_path = os.path.join(dirs['checkpoint_dir'], 'postprocess')
logger = TimeLogger()
#%%
# Load.
with open(os.path.join(dirs["session_dir"], 'session.pkl'), 'rb') as f:
    session = dill.load(f)
# with open(dirs["model_path"], 'rb') as f:
#     model = dill.load(f)
model = lutils.PLWrapper.load_from_config_path(dirs["config_path"]) #@TODO something is broken
with open(dirs["config_path"], 'r') as f:
    train_config = json.load(f)
    
cids = session['cids']
model = model.to(device)
model.eval()
model.model = model.model.to(device)
core = model.model.core if hasattr(model.model, 'core') else None
readout = model.model.readout if hasattr(model.model, 'readout') else None
c, x, y, num_lags = session['input_dims']
logger.log('Loaded model.')

#%%
ds = FixationMultiDataset.load(dirs["ds_dir"])
ds.use_blocks = False
val_inds = ds.block_inds_to_fix_inds(session["val_inds"])
val_dl = lutils.get_fix_dataloader(ds, val_inds, batch_size=1)
#%% 
# Evaluate model
model.loss = NDNT.metrics.poisson_loss.PoissonLoss_datafilter()
ev = utils.eval_model_summary(model, val_dl)
best_cids = ev.argsort()[::-1]
logger.log('Evaluated model.')

#%%
# zero irfs shape (neuron, time, 1, x, y)
zero_irfs = torch.stack(
    [
        utils.get_zero_irfC(
            (num_lags, c, x, y),
            model, cc, device
        ) for cc in tqdm(best_cids, desc='Calculating zero IRFs')
    ], dim=0
).squeeze(2)
zero_irfs = zero_irfs / torch.abs(zero_irfs).amax((1, 2, 3), keepdim=True)
irf_powers = zero_irfs.pow(2).mean((2, 3))
#plot irf powers for each neuron
plt.figure(figsize=(10, len(best_cids)))
plt.suptitle('IRF Powers')
for i in tqdm(range(len(best_cids)), desc='Plotting IRF Powers'):
    plt.subplot(len(best_cids), 1, i+1)
    plt.plot(irf_powers[best_cids[i]], label=best_cids[i])
    # plt.ylim(0, irf_powers.max())
    plt.legend(loc='upper right')
plt.tight_layout()

fig = utils.plot_grid(zero_irfs, vmin=-1, vmax=1, titles=best_cids, suptitle='Zero IRFs')
del zero_irfs, irf_powers
torch.cuda.empty_cache()
#%%    
try:
    is_ndnt_plottable = hasattr(core[0], 'get_weights') and readout is not None
except:
    is_ndnt_plottable = False
    
# Plot spatiotemporal first layer kernels.
if is_ndnt_plottable and not config["fast"]:
    w_normed = utils.zscoreWeights(core[0].get_weights()) #z-score weights per neuron for visualization
    plot_sta_movie(w_normed, frameDelay=1, path=tosave_path+'_weights2D.gif', cmap='viridis')
    # plot_sta_movie(w_normed, frameDelay=1, path=tosave_path+'_weights3D.gif', threeD=True, cmap='viridis')
    logger.log('Plotted first layer weight movies.')

#%%
#loader is cyclic single item loader
# Integrated gradients analysis. TODO
# grads = utils.get_integrated_grad_summary(model, core, loader, nsamples=100)
# utils.get_grad_summary(model, core, loader, device, cids, True)
# utils.get_grad_summary(model, core, loader, device, cids, False)
# logger.log('Computed integrated gradients.')

# %%
# Plot gradients.
# _ = plot_stas(np.transpose(grads[0], (2, 0, 1, 3)))
# _ = plot_stas(grads[1])
# logger.log('Plotted gradients.')

# %%
# Plot transients.
try:
    sta_true, sta_hat, _ = plot_transientsC(model, val_dl, cids, num_lags)
    logger.log('Plotted transients.')
except Exception as e:
    logger.log('Failed in transients: %s'%e)

#%%
logger.log('Plotting model.')
if is_ndnt_plottable:
    utils.plot_model(model.model)
else:
    i = 0
    for module in model.model.modules():
        # check if convolutional layer
        if issubclass(type(module), nn.modules.conv._ConvNd):
            w = module.weight.data.cpu().numpy()
            if len(w.shape) == 5:
                w = w.squeeze(1) # asume cin is 1
            w = w/np.abs(w).max((0, 2, 3), keepdims=True) # normalize
            # shape is (cout, cin, x, y)
            titles = ['cout %d'%i for i in range(w.shape[0])]
            utils.plot_grid(w, titles=titles, suptitle='Layer %d'%i, desc='Layer %d'%i, vmin=-1, vmax=1)
            i += 1
logger.log('Plotted model.')
# %%
with open(tosave_path+'.txt', 'w') as f:
    to_write = [
        'Scores: ' + json.dumps(ev.tolist()),
        'cids: ' + json.dumps(cids.tolist()),
        'Best neurons: ' + json.dumps(cids[best_cids].tolist()),
    ]
    f.write('\n'.join(to_write))
pdf_file = PdfPages(tosave_path + '.pdf')
for fig in [plt.figure(n) for n in plt.get_fignums()]:
    fig.savefig(pdf_file, format='pdf')
pdf_file.close()
logger.log('Saved results as pdf.')
#%%
if not config["fast"]:
    print('Animating IRFs.')
    win = 240*2 # length of window in frames
    offset = 10 # index of initial fixation
    nfixations = 50 # number of fixations to plot
    slice_ind = 20 # column index to plot
    ncids = 3 # number of neurons to plot (sorted by bits per spike)
    assert len(val_inds) > offset + nfixations, "hit end of dataset when animating IRFs"
    irf_dl = lutils.get_fix_dataloader(ds, val_inds[offset:offset+nfixations], batch_size=1)
    val_data_slice = ds[val_inds[offset:offset+nfixations]]
    stims, robs, dfs = val_data_slice["stim"], val_data_slice["robs"], val_data_slice["dfs"]
    fixation_ind = [ind for ind in val_inds[offset:offset+nfixations] for _ in range(len(ds[ind]["robs"]))]
    is_saccade = [False] + (np.diff(fixation_ind) != 0).tolist()
    movie_cid_list = list(best_cids[:ncids])
    cc_originals = [cids[cc] for cc in movie_cid_list]
    rfs, robs_hat = [[] for _ in movie_cid_list], [[] for _ in movie_cid_list]
    logger.log('Calculating IRFs over validation dataset.')
    for i, batch in tqdm(enumerate(irf_dl), desc='Batch', total=len(irf_dl)):
        batch = to_device(batch, device)
        outp = model(batch).cpu()
        for cc_ind in tqdm(range(len(movie_cid_list)), leave=False, desc='Neuron'):
            temp = irfC(batch, model, movie_cid_list[cc_ind], num_lags).cpu().reshape(len(batch["stim"])-num_lags+1, num_lags, x, y)[..., slice_ind]
            temp = torch.cat((torch.zeros(num_lags-1, num_lags, x), temp), dim=0)
            rfs[cc_ind].append(temp)
            robs_hat[cc_ind].append(outp[:, cc_originals[cc_ind]].detach().clone())
        del batch, outp
        torch.cuda.empty_cache()
    logger.log('Preparing IRF animations.')
    stims = stims[..., slice_ind].detach().cpu().numpy()
    robs_hat = torch.stack([torch.concat(i, dim = 0) for i in robs_hat], dim=1).detach().cpu().numpy()
    robs = robs[:, movie_cid_list].cpu().numpy()
    dfs = dfs[:, movie_cid_list].cpu().numpy()
    rfs = torch.stack([torch.concat(i, dim = 0) for i in rfs], dim=-1).squeeze(2).permute(0, 2, 1, 3)
    rfs = (rfs/torch.abs(rfs).amax((0, 1, 2), keepdim=True)).detach().cpu().numpy()
    for cc_ind, cc in tqdm(enumerate(movie_cid_list), desc='Animating IRFs'):
        cc_original = cc_originals[cc_ind]
        fig = plt.figure()
        plt.suptitle('Frame 0')
        plt.subplot(3, 1, 1)
        plt.title('Stimulus')
        im1 = plt.imshow(np.zeros((x, win)), cmap='viridis', aspect='auto', animated=True, vmin=-1, vmax=1)
        im1a = plt.gca()
        plt.yticks([])
        plt.xticks([])
        plt.subplot(3, 3, 6)
        plt.title('IRF')
        im2 = plt.imshow(np.zeros((x, num_lags)), cmap='viridis', aspect='auto', animated=True, vmin=-1, vmax=1)
        im2a = plt.gca()
        plt.yticks([])
        plt.xticks([])
        plt.subplot(3, 1, 3)
        plt.title('Response')
        x_vals = np.arange(1, win+1)
        pl1, = plt.plot(x_vals, np.zeros(win), label='true', c='blue')
        pl2, = plt.plot(x_vals, np.zeros(win), label='pred', c='red')
        pla = plt.gca()
        plt.ylim(np.min([robs[:, cc_ind].min(), robs_hat[:, cc_ind].min()]), np.max([robs[:, cc_ind].max(), robs_hat[:, cc_ind].max()]))
        plt.xlim(0, 23)
        plt.legend(loc='upper right')
        plt.tight_layout()

        saccade_id = np.where(is_saccade)[0]

        def animate(j):
            fig.suptitle(f'Neuron {cc_original} ({cc}th cid), Frame {j}')
            i = j + win
            im1.set_data(stims[j:i, :, 0].T)
            im2.set_data(rfs[i, :, ::-1, cc_ind])
            x_vals = np.arange(j, i)

            pla.set_xlim(j, i)
            pl1.set_data(
                x_vals,
                robs[j:i, cc_ind],
            )
            pl2.set_data(
                x_vals,
                robs_hat[j:i, cc_ind],
            )

            sacticks = saccade_id[np.logical_and(saccade_id > j, saccade_id < i)]
            im1a.set_xticks(sacticks-j)

            im1a.set_xticklabels([])
            pla.set_xticks(sacticks)
            pla.set_xticklabels([])
            return [im1, im2, pl1, pl2]

        fps = 30
        anim = FuncAnimation(
            fig,
            animate,
            frames = len(stims)-win,
            interval = int((len(stims)-win)/fps),
            blit=True
        )

        anim.save(tosave_path+'_video_cid%d.mp4'%cc_original, writer = animation.FFMpegWriter(fps=fps))
    del stims, robs, is_saccade, rfs, robs_hat, val_data_slice
    torch.cuda.empty_cache()
    logger.log('Animated IRFs.')
logger.closure()