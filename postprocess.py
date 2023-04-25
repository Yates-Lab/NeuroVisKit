'''
    Analysis of trained models.
'''
#%%
import torch
import json
import NDNT
import numpy as np
import dill
import os, sys, getopt, __main__
from utils.utils import unpickle_data, get_datasets, plot_transients, seed_everything, TimeLogger
from models.utils import plot_stas
import utils.postprocess as utils
from models.utils.plotting import plot_sta_movie
import matplotlib.pyplot as plt
from utils.loss import NDNTLossWrapper
from matplotlib.backends.backend_pdf import PdfPages
seed_everything(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
nsamples_train=1
nsamples_val=None #56643 | None is equivalent to all
batch_size=1000 # Reduce if you run out of memory
run_name = 'test_smooth'
session_name = '20200304'
isPytorch = False # skip NDNT-only utils
fast = False # skip making movies
sigmoid = False # force sigmoid activation

if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"d:b:o:n:s:pl:f",["device=", "batch_size=", "run_name=", "session_name=", "pytorch", "loss=", "fast", "sigmoid"])
    for opt, arg in opts:
        if opt in ("-d", "--device"):
            device = torch.device(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-n", "--run_name"):
            run_name = arg
        elif opt in ("-s", "--session_name"):
            session_name = arg
        elif opt in ("-p", "--pytorch"):
            isPytorch = True
        elif opt in ("-f", "--fast"):
            fast = True
        elif opt in ("--sigmoid"):
            sigmoid = True

#%%
# Determine paths.
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data', 'sessions', session_name)
model_path = os.path.join(cwd, 'data', 'models', run_name)
tosave_path = os.path.join(model_path, 'postprocess')
logger = TimeLogger()

#%%
# Load.
with open(os.path.join(data_path, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids = session['cids']
with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
    model = dill.load(f)
if os.path.exists(os.path.join(model_path, 'config.pkl')):
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = dill.load(f)
else:
    config = {}
model = model.to(device)
model.model = model.model.to(device)
if not isPytorch:
    core = model.model.core
input_dims = session['input_dims'][1:]
logger.log('Loaded model.')

#%%
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, device=device, path=data_path)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, batch_size=batch_size, device=device)
loader = utils.Loader(val_ds, shuffled=True, cyclic=True)
logger.log('Loaded data.')

#%% 
# Evaluate model and plot first layer.
isZScore = hasattr(model, "t_mean")
isNDNT = isinstance(model.loss, NDNTLossWrapper)
if not isNDNT:
    model.loss = NDNT.metrics.poisson_loss.PoissonLoss_datafilter()
if sigmoid:
    model.model.output_NL = torch.nn.Sigmoid()
if isZScore: # use t_mean and t_std to convert from z-score to firing rate
    if len(model.t_mean) != len(cids):
        model.t_mean = model.t_mean[session['cids']]
        model.t_std = model.t_std[session['cids']]
    ev = utils.eval_model_summary(model, val_dl, t_mean=model.t_std.to(device), t_std=model.t_std.to(device))
else:
    ev = utils.eval_model_summary(model, val_dl)
    
best_cids = ev.argsort()[::-1]
if not isPytorch:
    utils.plot_layer(core[0])
logger.log('Evaluated model.')

if isZScore:
    fsize = 5
    for i in range(len(val_ds)-1, fsize-1, -1):
        val_data["robs"][i] = val_data["robs"][i-fsize:i].mean(0)
    if isZScore:
        val_data["robs"] = (val_data["robs"] - val_data["robs"].mean(0, keepdims=True)) / val_data["robs"].std(0, keepdims=True)
elif "preprocess" in config and config["preprocess"] == "binarize":
    inds2 = torch.where(train_data['robs'] > 1)[0]
    train_data['robs'][inds2 - 1] = 1
    train_data['robs'][inds2 + 1] = 1
    train_data['robs'][inds2] = 1
    inds2 = torch.where(val_data['robs'] > 1)[0]
    val_data['robs'][inds2 - 1] = 1
    val_data['robs'][inds2 + 1] = 1
    val_data['robs'][inds2] = 1
#%%
zero_irfs = []
for cc in best_cids:
    z = utils.get_zero_irf(input_dims, model, cc, device).permute(2, 0, 1)
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
offset = 54137 # index of initial frame
movie_cid_list = [] if fast else best_cids[:3]
for cc in movie_cid_list:
    stims = val_data["stim"][offset:offset+n+win].reshape(-1, *input_dims).squeeze()
    robs = val_data["robs"][offset:offset+n+win, cc]
    is_saccade = val_data["fixation_num"][offset:offset+n+win] != val_data["fixation_num"][offset-1:offset-1+n+win]
    rfs = irf({"stim": stims}, model, cc)
    probs = model({"stim": stims})[:, cc]
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
        fig.suptitle(f'Neuron {cc}, Frame {j}')
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

    anim.save(tosave_path+'_video_cid%d.mp4'%cc, writer = animation.FFMpegWriter(fps=fps))
    del stims, robs, is_saccade, rfs, probs
#%%    
# Plot spatiotemporal first layer kernels.
if not isPytorch:
    w_normed = utils.zscoreWeights(core[0].get_weights()) #z-score weights per neuron for visualization
    plot_sta_movie(w_normed, frameDelay=1, path=tosave_path+'_weights2D.gif', cmap='viridis')
    # plot_sta_movie(w_normed, frameDelay=1, path=tosave_path+'_weights3D.gif', threeD=True, cmap='viridis')
    logger.log('Plotted first layer weight movies.')

#%%
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
sta_true, sta_hat, _ = plot_transients(model, val_data, device="cpu")
# sta_true, sta_hat (n saccades, n timepoints, n neurons)
logger.log('Plotted transients.')

#%%
if not isPytorch:
    utils.plot_model(model.model)
    logger.log('Plotted model.')

# %%
with open(tosave_path+'.txt', 'w') as f:
    to_write = [
        'Scores: ' + json.dumps(ev.tolist()),
        'Best neurons: ' + json.dumps(best_cids.tolist()),
    ]
    f.write('\n'.join(to_write))
pdf_file = PdfPages(tosave_path + '.pdf')
for fig in [plt.figure(n) for n in plt.get_fignums()]:
    fig.savefig(pdf_file, format='pdf')
pdf_file.close()
logger.log('Saved results as pdf.')
# %%
