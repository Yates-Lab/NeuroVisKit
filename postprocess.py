'''
    Analysis of trained models.
'''
#%%
import torch
import json
import numpy as np
import dill
import os, sys, getopt
from utils.utils import unpickle_data, get_datasets, plot_transients, seed_everything, TimeLogger
from models.utils import plot_stas
import utils.postprocess as utils
from models.utils.plotting import plot_sta_movie
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
seed_everything(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
nsamples_train=1
nsamples_val=None #56643
batch_size=1000 # Reduce if you run out of memory
run_name = 'test'
session_name = '20200304'

if __name__ == "__main__" and "IPython" not in sys.modules:
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"d:b:o:n:s:",["device=", "batch_size=", "run_name=", "session_name="])
    for opt, arg in opts:
        if opt in ("-d", "--device"):
            device = torch.device(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-n", "--run_name"):
            run_name = arg
        elif opt in ("-s", "--session_name"):
            session_name = arg

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
model.to(device)
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
ev = utils.eval_model_summary(model, val_dl)
best_cids = ev.argsort()[::-1]
utils.plot_layer(core[0])
logger.log('Evaluated model.')

#%%    
# Plot spatiotemporal first layer kernels.
w_normed = utils.zscoreWeights(core[0].get_weights()) #z-score weights per neuron for visualization
plot_sta_movie(w_normed, frameDelay=1, path=tosave_path+'_weights2D.gif', cmap='cool')
plot_sta_movie(w_normed, frameDelay=1, path=tosave_path+'_weights3D.gif', threeD=True, cmap='cool')
logger.log('Plotted first layer + movies.')

#%%
# Integrated gradients analysis. TODO
grads = utils.get_integrated_grad_summary(model, core, loader, nsamples=100)
utils.get_grad_summary(model, core, loader, device, cids, True)
utils.get_grad_summary(model, core, loader, device, cids, False)
logger.log('Computed integrated gradients.')

# %%
# Plot gradients.
_ = plot_stas(np.transpose(grads[0], (2, 0, 1, 3)))
_ = plot_stas(grads[1])
logger.log('Plotted gradients.')

# %%
# Plot transients.
sta_true, sta_hat, _ = plot_transients(model, val_data)
logger.log('Plotted transients.')

#%%
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