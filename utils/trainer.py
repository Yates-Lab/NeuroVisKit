'''
    Function-Oriented Trainer
'''

import os
import dill
import numpy as np
import torch
import traceback
from tqdm import tqdm  # progress bar
import matplotlib.pyplot as plt
# TODO -> identify the epoch of val_loss minimum
# TODO -> get val data on device

def train(model,
                train_loader,
                val_loader,
                optimizer=None,
                optimize_graph=False,
                max_epochs=100,
                verbose=0,
                seed=0,
                patience=10,
                checkpoint_path=None,
                device="cuda",
                plot=False,
                memory_saver=False,
                ):
    '''
        Parameters
        ----------

        verbose : int
            Determines feedback.
            - 0 for no feedback
            - 1 for minimal feedback such as error reporting
            - 2+ full feedback
        patience : int/None
            Determines if there is early stopping, and how many epochs of patience (decreases if val >= min_val).
        checkpoint_path : str/None
            Checkpoints the model with best validation score (entire model).
        plot : bool
            Plots validation and train score over time
    '''
    
    n_iter = 0
    val_loss_min = np.Inf
    torch.cuda.empty_cache()
    if optimize_graph:
        torch.backends.cudnn.benchmark = True
    if seed is not None:
        # set seed only for cuda
        torch.cuda.manual_seed(seed)

    def verbose_print(*args, v=0, **kwargs):
        # print if verbose level is at least v
        if v <= verbose:
            print(*args, **kwargs)

    def train_one_step(data):
        nonlocal n_iter
        # for dsub in data:
        #     if data[dsub].device != device:
        #         data[dsub] = data[dsub].to(device)
        out = model.training_step(data)
        if memory_saver:
            for dsub in data:
                data[dsub] = data[dsub].cpu()
        n_iter += 1
        loss = out['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return {'train_loss': loss.detach().item()}

    def train_one_epoch(train_loader):
        model.train()  # set model to training mode
        runningloss = 0
        out_size = len(train_loader)
        iterator = tqdm if verbose > 1 else lambda x: x
        for data in iterator(train_loader):
            out = train_one_step(data)
            if np.isnan(out['train_loss']):
                print("nan training loss.")
                break
            runningloss += out['train_loss']
            torch.cuda.empty_cache()
        # should this be an aggregate out?
        return {'train_loss': runningloss/out_size}

    def validate_one_epoch(val_loader):
        model.eval()
        runningloss = 0
        nsteps = len(val_loader)
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                for dsub in data:
                    if data[dsub].device != device:
                        data[dsub] = data[dsub].to(device)
                out = model.validation_step(data)
                if memory_saver:
                    for dsub in data:
                        data[dsub] = data[dsub].cpu()
                runningloss += out['val_loss'].item()
                torch.cuda.empty_cache()
        return {'val_loss': runningloss/nsteps}

    train_scores, val_scores = [], []
    def graceful_exit(reason=""):
        verbose_print(f"Exitting training...{reason}", v=1)
        model.to("cpu")
        torch.cuda.empty_cache()
        if plot and len(train_scores) > 0:
            print("attempted plotting")
            plt.figure()
            plt.plot(np.arange(len(train_scores)), train_scores, label="Train Loss", c="b")
            plt.plot(np.arange(len(val_scores)), val_scores, label="Val Loss", c="r")
            plt.legend()
        else:
            print("no plotting", train_scores, val_scores, plot)
        return model.eval()

    optimizer.zero_grad()
    # main loop for training
    try:
        for epoch in range(max_epochs):
            if hasattr(model, "pre_epoch"):
                model.pre_epoch()
            # train one epoch
            out = train_one_epoch(train_loader)
            train_loss = out['train_loss']
            if np.isnan(train_loss):
                verbose_print("Exiting: nan training loss.", v=1)
                break
            out = validate_one_epoch(val_loader)
            this_val = out['val_loss']
            train_scores.append(train_loss)
            val_scores.append(this_val)
            if val_loss_min > this_val:
                verbose_print("Checkpointing model...", v=2)
                if checkpoint_path is not None:
                    torch.save(model, os.path.join(checkpoint_path, 'model.pkl'))
                verbose_print("Finished checkpointing", v=2)
            elif patience is not None:
                patience -= 1
                verbose_print(
                    f"Losing patience ಠ_ಠ (best val: {val_loss_min:.4f}, current val: {this_val:.4f}, patience: {patience})", v=2)
                if patience <= 0:
                    verbose_print("Early stopping...", "epoch", epoch, v=1)
                    break
            val_loss_min = min(this_val, val_loss_min)
            verbose_print("Epoch %d: train loss %.4f val loss %.4f" %
                        (epoch, train_loss, this_val), v=2)
    except (KeyboardInterrupt, Exception):
        print("Exception caught:")
        print(traceback.format_exc())
        print("End of exception, exiting gracefully.")
    graceful_exit()
    print(val_loss_min)
    return val_loss_min
