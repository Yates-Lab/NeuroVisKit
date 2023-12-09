import torch.nn as nn
# from models import ModelWrapper
# from models import CNNdense as CNNdenseNDNT
from NeuroVisKit.utils import regularization
from NeuroVisKit.utils.loss import Poisson
# from unet import UNet, BioV
# from models.cnns import DenseReadout as DenseReadoutNDNT

class ModelWrapper(nn.Module):
    '''
    Base class for model wrappers that allows for easy regularization and loss computation.
    '''
    def __init__(self,
            model, # the model to be trained
            loss=Poisson(), # the loss function to use
            cids = None, # which units to use during fitting
            **kwargs,
            ):
        
        super().__init__()
        
        if cids is None:
            self.cids = model.cids
        else:
            self.cids = cids
        
        self.model = model
        if hasattr(model, 'name'):
            self.name = model.name
        else:
            self.name = 'unnamed'

        self.loss = loss
    def compute_reg_loss(self):
        return self.model.compute_reg_loss()
    def prepare_regularization(self, normalize_reg = False):
        self.model.prepare_regularization(normalize_reg=normalize_reg)
    def forward(self, batch):
        return self.model(batch)
    def training_step(self, batch, batch_idx=None, alternative_loss_fn=None):  # batch_indx not used, right?
        y = batch['robs'][:,self.cids]
        y_hat = self(batch)

        if alternative_loss_fn is None:
            if 'dfs' in batch.keys():
                dfs = batch['dfs'][:,self.cids]
                loss = self.loss(y_hat, y, dfs)
            else:
                loss = self.loss(y_hat, y)
        else:
            loss = alternative_loss_fn(y_hat, batch)

        regularizers = self.compute_reg_loss()

        return {'loss': loss.sum() + regularizers, 'train_loss': loss.mean(), 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]
        
        y_hat = self(batch)

        if 'dfs' in batch.keys():
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}

class PytorchWrapper(ModelWrapper):
    """Model wrapper meant to be used with native pytorch models.

    Args:
        ModelWrapper (_type_): _description_
    """
    def __init__(self, model, *args, cids=None, bypass_preprocess=False, **kwargs):
        super().__init__(model, *args, cids=cids, **kwargs)
        self.bypass_preprocess = bypass_preprocess
        if not hasattr(self, 'cids'):
            self.cids = cids
        self.reg = regularization.extract_reg(self.model, proximal=False)
        self.proximal_reg = regularization.extract_reg(self.model, proximal=True)
        self._lr = None
        print("initialized", self.reg, self.proximal_reg)
    def train(self, mode=True):
        self.reg = regularization.extract_reg(self.model, proximal=False)
        self.proximal_reg = regularization.extract_reg(self.model, proximal=True)
        return super().train(mode)
    def compute_reg_loss(self, *args, **kwargs):
        loss = sum([r() for r in self.reg]+[0.0])
        return loss
    def compute_proximal_reg_loss(self, *args, **kwargs):
        return sum([r() for r in self.proximal_reg]+[0.0])
    def forward(self, x, pass_dict=False, *args, **kwargs):
        if not self.bypass_preprocess:
            if type(x) is dict and not pass_dict:
                x = x['stim']
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim == 4:
                x = x.unsqueeze(1)
        return self.model(x, *args, **kwargs)
    @property
    def lr(self):
        return self._lr
    @lr.setter
    def lr(self, lr):
        self._lr = lr
        for i in self.proximal_reg:
            if hasattr(i, 'lr'):
                i.lr = lr