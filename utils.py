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
