#%%
import os, sys, shutil, dill
import torch
from weakref import proxy
from NeuroVisKit._utils.io import dump
from lightning.pytorch.callbacks import ModelCheckpoint

class ThoroughModelCheckpoint(ModelCheckpoint):
    """A ModelCheckpoint that saves the entire model without dependencies.
    **WARNING: may result in very large files.**
    """
    def _save_checkpoint(self, trainer, filepath) -> None:
        #make any dirs if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        out = trainer.lightning_module.pre_save() if hasattr(trainer.lightning_module, 'pre_save') else trainer.lightning_module
        dump(out, filepath)
        # trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

def pl_device_format(device):
    """Convert a torch device to the format expected by pytorch lightning.
    **ONLY WORKS FOR SINGLE DEVICE**

    Args:
        device: str or device object describing which device to use

    Returns:
        str: the device formatted for pytorch lightning
    """
    if type(device) == torch.device:
        device = str(device)
    if "cpu" in device:
        return 'cpu'
    if type(device) == str:
        return ",".join(device.split("cuda:"))[1:] + ','

def get_dirnames(config):
    """Returns a dictionary of directory names for the given config.
    Args:
        config (dict): the config dictionary
    Returns:
        dict: a dictionary of directory names
    """
    return {
        'dirname': config["dirname"],
        'checkpoint_dir': os.path.join(config["dirname"], 'models', config['name']),
        'session_dir': os.path.join(config["dirname"], 'sessions', config["session"]),
        'model_path': os.path.join(config["dirname"], 'models', config['name'], 'model.pkl'),
        'config_path': os.path.join(config["dirname"], 'models', config['name'], 'config.json'),
        'ds_dir': os.path.join(config["dirname"], 'sessions', config["session"], 'ds.pkl'),
        'log_dir': os.path.join(config["dirname"], 'models', config['name'], 'lightning_logs'),
    }
    
def prepare_dirs(config):
    """Creates the directories for the given config.

    Args:
        config (dict): the config dictionary
    Returns:
        dirs (dict): a dictionary of directory names
        config (dict): the config dictionary that has been updated with the session metadata.
        session (dict): the session dictionary containing metadata for the intended dataset.
    """
    overwrite = config['overwrite']
    from_checkpoint = config['from_checkpoint']
    print('Device: ', config['device'])
    dirs = get_dirnames(config)
    if os.path.exists(dirs['checkpoint_dir']) and not overwrite and not from_checkpoint:
        print('Directory already exists. Exiting.')
        print('If you want to overwrite, use the -o flag.')
        sys.exit()
    elif overwrite and not from_checkpoint:
        if os.path.exists(dirs['checkpoint_dir']):
            shutil.rmtree(dirs['checkpoint_dir'])
        else:
            print('Directory does not exist (did not overwrite).')
    os.makedirs(dirs['checkpoint_dir'], exist_ok=True)   
    # os.makedirs(dirs['log_dir'], exist_ok=True)
    with open(os.path.join(dirs["session_dir"], 'session.pkl'), 'rb') as f:
        session = dill.load(f)
    config.update({
        'cids': session['cids'],
        'input_dims': session['input_dims'],
        'mu': session['mu'],
    })
    return dirs, config, session