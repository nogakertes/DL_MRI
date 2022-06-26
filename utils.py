import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import *
from data_loader import loadFromDir, showKspaceFromTensor
from torch.nn import MSELoss
from torch.optim import Adam, SGD
import torch.nn.functional as F
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
import fastmri
import config


def save_model(model, models_path, ep=config.EPOCHS):
    path = os.path.join(models_path, f'model_ep_{ep}.pth')
    print(f'Saved model from epoch {ep} at {path}')
    torch.save(model, path)

def init_paths(base_results_path, base_models_path, curr_model):
    # Set paths infrastructure
    models_path = os.path.join(base_models_path, curr_model)
    if not os.path.exists(models_path):
        raise Exception(f'Error! {curr_model} models directory does not exist!')

    results_path = os.path.join(base_results_path, curr_model)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return models_path


def get_best_model_ep(models_path):
    # Find the best model
    best_epoch_model = 0
    for file in os.listdir(models_path):
        model_epoch = os.path.split(file)[1].split('.')[0].split('_')[-1]
        if int(model_epoch) >= best_epoch_model:
            best_epoch_model = int(model_epoch)

    return best_epoch_model


