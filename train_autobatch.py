'''
    Script for generating a nice fitting pipeline.
'''
#%%
import os, sys, getopt, __main__, math
import json
import torch
import dill
from utils.utils import seed_everything, unpickle_data, get_datasets
import utils.train as utils
from utils.loss import get_loss, DatafilterLossWrapper
from utils.get_models import get_model
import statsmodels.api as sm
from scipy.stats import shapiro, kstest, norm
from scipy.optimize import curve_fit
from statsmodels.stats.diagnostic import lilliefors
from scipy.signal import convolve
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import bisect
seed_everything(0)

run_name = 'test' # Name of log dir.
session_name = '20200304'
nsamples_train=236452
nsamples_val=56643
overwrite = False
from_checkpoint = False
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
seed = 420
config = {
    'loss': 'poisson', # utils/loss.py for more loss options
    'model': 'CNNdense', # utils/get_models.py for more model options
    'trainer': 'adam', # utils/train.py for more trainer options
    'filters': [20, 20, 20, 20], # the config is fed into the model constructor
    'kernels': [11, 11, 11, 11],
    'preprocess': None,#'binarize', # this further preprocesses the data before training
    'override_output_NL': True, # this overrides the output nonlinearity of the model according to the loss
}

# Here we make sure that the script can be run from the command line.
if __name__ == "__main__" and not hasattr(__main__, 'get_ipython'):
    argv = sys.argv[1:]
    opt_names = ["name=", "seed=", "train=", "val=", "config=", "from_checkpoint", "overwrite", "device=", "session=", "loss=", "model=", "trainer=", "preprocess=", "override_NL"]
    opts, args = getopt.getopt(argv,"n:s:oc:d:l:m:t:p:", opt_names)
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
            
dirname = os.path.join(os.getcwd(), 'data')
checkpoint_dir = os.path.join(dirname, 'models', run_name)
data_dir = os.path.join(dirname, 'sessions', session_name)
with open(os.path.join(dirname, 'sessions', session_name, 'session.pkl'), 'rb') as f:
    session = dill.load(f)
cids = session['cids']
# %%
# Load data.
train_data, val_data = unpickle_data(nsamples_train=nsamples_train, nsamples_val=nsamples_val, path=data_dir)
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=device)

with open(os.path.join(checkpoint_dir, 'model.pkl'), 'rb') as f:
    model = dill.load(f).to(device)
#%%
# Load model and preprocess data.

def poisson_f(pred, target, reduce=False, *args, **kwargs):
    return F.poisson_nll_loss(pred, target, log_input=False, full=False, reduction="mean" if reduce else "none")

with torch.no_grad():
    processed = torch.concat([model({"stim": i["stim"]}) for i in tqdm(torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=True))], 0)
    processed = (poisson_f(processed, train_ds.covariates["stim"][:, cids]) * train_ds.covariates["dfs"][:, cids]).sum(1)/train_ds.covariates["dfs"][:, cids].sum(1)
processed = processed.cpu().numpy()

#%%

metrics = {
    'shapiro': lambda x: -shapiro(x)[0],
    # 'lilliefors': lambda x: lilliefors(x)[0],
    'ks': lambda x: kstest(x, 'norm', args=norm.fit(x))[0],
}

def test_batch_sizes(sizes, metric, plot=True, smooth=False):
    vals = []
    for i in sizes:
        valid_data = [processed[j*i:(j+1)*i].sum() for j in range(int(len(processed)//i))]
        vals.append(metrics[metric](valid_data))
    if plot:
        plt.figure()
        plt.title(metric)
        if smooth:
            smooth = smooth if isinstance(smooth, int) else 10
            means = convolve(vals, np.ones(smooth)/smooth, mode='same')
            stds = np.sqrt(convolve((vals-means)**2, np.ones(smooth)/smooth, mode='same'))
            plt.plot(sizes, means, linewidth=1)
            plt.fill_between(sizes, means-stds, means+stds, alpha=0.5)
            plt.ylim([np.min((means-stds)[smooth:-smooth]), np.max((means+stds)[smooth:-smooth])])
            plt.legend(["mean", "std"])
        else:
            plt.plot(sizes, vals)
    return sizes[np.argmin(vals)]

def scan_search_bsizes(metric, max=1000):
    return test_batch_sizes(np.arange(1, max), metric, smooth=10)
    
def bisection_search_bsizes(metric, max=15000):
    def searcher(search_sizes, metric):
        search = test_batch_sizes(search_sizes, metric)
        diffs = search_sizes-search
        low_ind = np.where(diffs < 0)[0][diffs[diffs < 0].argmax()]
        high_ind = np.where(diffs > 0)[0][diffs[diffs > 0].argmin()]
        return low_ind, high_ind

    search_sizes = 2**np.arange(int(math.log2(max)))
    low_ind, high_ind = searcher(search_sizes, metric)
    while search_sizes[high_ind]-search_sizes[low_ind] > 100:
        search_sizes = np.arange(search_sizes[low_ind], search_sizes[high_ind], (search_sizes[high_ind]-search_sizes[low_ind])//100)
        low_ind, high_ind = searcher(search_sizes, metric)
    search_sizes = np.arange(search_sizes[low_ind], search_sizes[high_ind], 1)
    return test_batch_sizes(search_sizes, metric)

for i in metrics.keys():
    print(i, scan_search_bsizes(i))
# #%%

# def plot_qq(x, ax=None):
#         sm.qqplot(x, line='s', ax=ax, fit=True)
#         # plt.title("Q-Q Plot")
#         # plt.xlabel("Theoretical Quantiles")
#         # plt.ylabel("Sample Quantiles")
#         plt.title("")
#         plt.xlabel("")
#         plt.ylabel("")
# # def evaluate_normal_fit(data):
# #     # Fit the data to a normal distribution
# #     mu_hat, sigma_hat = norm.fit(data)

# #     # Calculate the Kolmogorov-Smirnov test statistic
# #     ks_stat, p_value = kstest(data, 'norm', args=(mu_hat, sigma_hat))

# #     # Return the Kolmogorov-Smirnov statistic
# #     return ks_stat
# n = 15
# plt.figure(figsize=(25, n))
# for i in range(n):
#     valid_data = convolve(processed, torch.ones(2**i).numpy()/2**i, mode='valid')
#     valid_data = [processed[j*2**i:(j+1)*2**i].sum() for j in range(int(len(processed)//2**i))]
#     # if len(valid_data) > 4999:
#     #     valid_data = np.random.choice(valid_data, 4999, replace=False)
#     plt.subplot(int(n/5), 5, i+1)
#     plt.hist(valid_data, bins=100)
#     # plot_qq(valid_data, ax=plt.gca())
#     skew = np.mean((valid_data - np.mean(valid_data))**3) / np.mean((valid_data - np.mean(valid_data))**2)**(3/2)
#     shap = shapiro(valid_data)[0]
#     lili = lilliefors(valid_data)[0]
#     ks = kstest(valid_data, 'norm')[0]
#     plt.title(f"{shap:.3} | {lili:.3} | {ks:.3} | {skew:.3}")
# plt.tight_layout()

# plt.figure(figsize=(25, n))
# for i in range(n):
#     # valid_data = convolve(processed, torch.ones(2**i).numpy()/2**i, mode='valid')
#     valid_data = np.array([processed[j*2**i:(j+1)*2**i].sum() for j in range(int(len(processed)//2**i))])
#     # if len(valid_data) > 4999:
#     #     valid_data = np.random.choice(valid_data, 4999, replace=False)
#     plt.subplot(int(n/5), 5, i+1)
#     # plt.hist(valid_data, bins=100)
#     plot_qq(valid_data, ax=plt.gca())
#     skew = np.mean((valid_data - np.mean(valid_data))**3) / np.mean((valid_data - np.mean(valid_data))**2)**(3/2)
#     shap = shapiro(valid_data)[0]
#     lili = lilliefors(valid_data)[0]
#     ks = kstest(valid_data, 'norm')[0]
#     plt.title(f"{shap:.3} | {lili:.3} | {ks:.3} | {skew:.3}")
# plt.tight_layout()
# # def test_batch_size():
# #     data = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=10000, shuffle=True)))
# #     with torch.no_grad():
# #         processed = model(data)
# #         samplewise_loss = model.loss.loss
# #         def poisson_f(pred, target, reduce=False, *args, **kwargs):
# #             return F.poisson_nll_loss(pred, target, log_input=False, full=False, reduction="mean" if reduce else "none")
# #         loss = poisson_f(processed, data["robs"][:, cids])
# #         data["dfs"][:, cids][~torch.isfinite(loss)] = 0
# #         sums = (loss*data["dfs"][:, cids]).sum(1) / data["dfs"][:, cids].sum(1)
# #         plt.figure()
# #         for i in [1, 10, 100, 1000, 10000]:
# #             valid_data = convolve(sums.cpu().numpy(), torch.ones(i).numpy()/i, mode='valid')
# #             def plot_qq(x, ax=None):
# #                 sm.qqplot(x, line='s', ax=ax, fit=True)
# #                 # plt.title("Q-Q Plot")
# #                 # plt.xlabel("Theoretical Quantiles")
# #                 # plt.ylabel("Sample Quantiles")
# #                 plt.title("")
# #                 plt.xlabel("")
# #                 plt.ylabel("")
# #             plt.subplot(1, 5, int(math.log10(i)))
# #             plot_qq(valid_data, ax=plt.gca())
# #             shap = shapiro(valid_data)[1]
# #             plt.title(f"{shap:.3}")
# #         plt.tight_layout()
            
#     # # now use the plot_qq function to plot the qq plot for each cell
#     # nrows = int(math.ceil(math.sqrt(cids.shape[0])))
#     # ncols = int(math.ceil(cids.shape[0] / nrows))
#     # plt.figure(figsize=(2*nrows, 2*ncols))
#     # for i, cid in enumerate(cids):
#     #     valid_data = loss[:, i][torch.isfinite(loss[:, i])].cpu().numpy()
#     #     plt.subplot(nrows, ncols, i+1)
#     #     plt.hist(loss[:, i].cpu().numpy())
#     #     # plot_qq(valid_data, ax=plt.gca())
#     #     shap = shapiro(valid_data)[1]
#     #     plt.title(f"{shap:.3}")
#     # plt.tight_layout()
#     # plt.figure()
#     # valid_data = loss[torch.isfinite(loss)].cpu().numpy().flatten()
#     # plot_qq(valid_data, ax=plt.gca())
#     # shap = shapiro(valid_data)[1]
#     # plt.title(f"{shap:.3}")
    
    
# %%
