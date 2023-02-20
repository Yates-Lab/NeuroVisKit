import torch
import dill
from utils.trainer import train
from models import ModelWrapper, CNNdense
from utils.get_models import get_cnn
from utils.utils import seed_everything

def train_lbfgs(model, train_loader, val_loader, checkpoint_dir, device=None):
    pass

def train_adam(model, train_loader, val_loader, checkpoint_dir, device):
    max_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    val_loss_min = train(
        model.to(device),
        train_loader,
        val_loader,
        optimizer=optimizer,
        max_epochs=max_epochs,
        verbose=2,
        checkpoint_path=checkpoint_dir,
        device=device,
        patience=30)
    return val_loss_min

TRAINER_DICT = {
    'adam': train_adam,
    'lbfgs': train_lbfgs,
}

def get_trainer(config):
    return TRAINER_DICT[config['trainer']]