'''
    Analysis of trained models.
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import NDNT
import numpy as np
import dill
import os, sys, getopt, __main__
from utils.utils import unpickle_data, get_datasets, plot_transients, seed_everything, TimeLogger, get_opt_dict, uneven_tqdm
from models.utils import plot_stas
import utils.postprocess as utils
import utils.lightning as lutils
from models.utils.plotting import plot_sta_movie
import matplotlib.pyplot as plt
from utils.loss import NDNTLossWrapper
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from datasets.mitchell.pixel.fixation import FixationMultiDataset
seed_everything(0)

config = {
    'batch_size': 3,
    'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
    'session': '20200304C',
    'name': 'testC_lr_new',
    'pytorch': False,
    'fast': False,
    'load_preprocessed': False,
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    loaded_opts = get_opt_dict([
        ('n:', 'name='),
        ('d:', 'device=', torch.device),
        ('s:', 'session='),
        ('p', 'pytorch'),
        ('f', 'fast'),
        ('b:', 'batch_size=', int),
        ('l', 'load_preprocessed'),
    ])
    config.update(loaded_opts)

#%%
# Determine paths.
device = config['device']
print("using device", str(device))
config["dirname"] = os.path.join(os.getcwd(), 'data')
dirs = lutils.get_dirnames(config)
tosave_path = os.path.join(dirs['checkpoint_dir'], 'postprocess')
logger = TimeLogger()

#%%
# Load.
with open(os.path.join(dirs["session_dir"], 'session.pkl'), 'rb') as f:
    session = dill.load(f)
# with open(dirs["model_path"], 'rb') as f:
#     model = dill.load(f)
model = lutils.PLWrapper.load_from_config_path(dirs["config_path"])
with open(dirs["config_path"], 'r') as f:
    train_config = json.load(f)
    
cids = session['cids']
model = model.to(device)
model.model = model.model.to(device)
if not config['pytorch']:
    core = model.model.core
input_dims = session['input_dims'][1:]
logger.log('Loaded model.')

#%%
if config['load_preprocessed']:
    with open(os.path.join(dirs["session_dir"], 'preprocessed.pkl'), 'rb') as f:
        ds = dill.load(f)
else:
    ds = FixationMultiDataset.load(dirs["ds_dir"])
    ds.use_blocks = True
if train_config["trainer"] == "lbfgs":
    train_dl = iter([ds[session["train_inds"]]])
    val_dl = iter([ds[session["val_inds"]]])
else:
    train_dl = lutils.get_fix_dataloader_preload(ds, session["train_inds"], batch_size=config['batch_size'], device=device)
    val_dl = lutils.get_fix_dataloader_preload(ds, session["val_inds"], batch_size=config['batch_size'], device=device)
#%% 
# Evaluate model and plot first layer.
isNDNT = isinstance(model.loss, NDNTLossWrapper)
if not isNDNT:
    model.loss = NDNT.metrics.poisson_loss.PoissonLoss_datafilter()
ev = utils.eval_model_summary(model, val_dl)
best_cids = ev.argsort()[::-1]
if not config["pytorch"]:
    utils.plot_layer(core[0])
logger.log('Evaluated model.')

#%%
try:
    # check if model requires flattened input
    try:
        utils.get_zero_irf(input_dims, model, 0, device)
        model_input_shape = input_dims
    except Exception as e:
        model_input_shape = (np.prod(input_dims),)
        
    zero_irfs = []
    for cc in best_cids:
        z = utils.get_zero_irf(model_input_shape, model, cc, device).reshape(input_dims).permute(2, 0, 1)
        zero_irfs.append(z / torch.amax(torch.abs(z)))
    irf_powers = torch.stack(zero_irfs).pow(2).mean((1, 2))
    #plot irf powers for each neuron
    plt.figure(figsize=(10, len(best_cids)))
    plt.suptitle('IRF Powers')
    for i in range(len(best_cids)):
        plt.subplot(len(best_cids), 1, i+1)
        plt.plot(irf_powers[best_cids[i]], label=best_cids[i])
        plt.ylim(0, irf_powers.max())
        plt.legend(loc='upper right')
    plt.tight_layout()

    fig = utils.plot_grid(zero_irfs, vmin=-1, vmax=1, titles=best_cids, suptitle='Zero IRFs')
    del zero_irfs
    #%%
    from matplotlib.animation import FuncAnimation
    from utils.mei import irf
    import matplotlib.animation as animation

    n = 240*10 # length of movie in frames
    win = 240*2
    offset = min(40000, len(val_data["stim"])-n-win)#44137 # index of initial frame
    movie_cid_list = [] if config["fast"] else best_cids[:3]
    for cc in movie_cid_list:
        cc_original = cids[cc]
        stims = val_data["stim"][offset:offset+n+win].reshape(-1, *input_dims).squeeze()
        robs = val_data["robs"][offset:offset+n+win, cc_original]
        is_saccade = val_data["fixation_num"][offset:offset+n+win] != val_data["fixation_num"][offset-1:offset-1+n+win]
        rfs = irf({"stim": stims.reshape(-1, *model_input_shape)}, model, cc).reshape(-1, *input_dims).squeeze()
        probs = model({"stim": stims.reshape(-1, *model_input_shape)})[:, cc]
        stims = stims[:, 20, :, :]
        rfs = rfs[:, 20, :, :]
        rfs = rfs / torch.amax(torch.abs(rfs), keepdim=True, axis=(1, 2))

        stims = stims.detach().cpu().numpy()
        robs = robs.detach().cpu().numpy()
        is_saccade = is_saccade.detach().cpu().numpy()
        rfs = rfs.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()

        fig = plt.figure()
        plt.suptitle('Frame 0')
        plt.subplot(3, 1, 1)
        plt.title('Stimulus')
        im1 = plt.imshow(np.zeros((35, win)), cmap='viridis', aspect='auto', animated=True, vmin=-1, vmax=1)
        im1a = plt.gca()
        plt.yticks([])
        plt.xticks([])
        plt.subplot(3, 3, 6)
        plt.title('IRF')
        im2 = plt.imshow(np.zeros((35, 24)), cmap='viridis', aspect='auto', animated=True, vmin=-1, vmax=1)
        im2a = plt.gca()
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

        saccade_id = np.where(is_saccade)[0]

        def animate(j):
            fig.suptitle(f'Neuron {cc_original} ({cc}th cid), Frame {j}')
            i = j + win
            im1.set_data(stims[j:i, :, 0].T)
            im2.set_data(rfs[i, :, ::-1])
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
            frames = n,
            interval = int(n/fps),
            blit=True
        )

        anim.save(tosave_path+'_video_cid%d.mp4'%cc_original, writer = animation.FFMpegWriter(fps=fps))
        del stims, robs, is_saccade, rfs, probs
except Exception as e:
    logger.log('Failed in IRFs: %s'%e)
#%%    
# Plot spatiotemporal first layer kernels.
if not config["pytorch"] and not config["fast"]:
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
    sta_true, sta_hat, _ = plot_transients(model, ds[session["val_inds"]], device="cpu")
    # sta_true, sta_hat (n saccades, n timepoints, n neurons)
    logger.log('Plotted transients.')
except Exception as e:
    logger.log('Failed in transients: %s'%e)

#%%
if not config["pytorch"] and hasattr(model.model, 'readout'):
    utils.plot_model(model.model)
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
# %%
