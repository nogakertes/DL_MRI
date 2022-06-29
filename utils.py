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


'''
This file contains different utilities and fuctions that we wrote and used more than once
'''


def save_model(model, models_path, ep=config.EPOCHS):
    '''
    This function saves the model to the model path and names it with it's epoch number
    '''
    path = os.path.join(models_path, f'model_ep_{ep}.pth')
    print(f'Saved model from epoch {ep} at {path}')
    torch.save(model, path)


def init_paths(base_results_path, base_models_path, curr_model):
    '''
    This fucntion initializes paths for evaluation:
    checks whether there is a models path, and raises exception if not.
    if the models path is valid, it creates a results path.
    returns the model path
    '''
    # Set paths infrastructure
    models_path = os.path.join(base_models_path, curr_model)
    if not os.path.exists(models_path):
        raise Exception(f'Error! {curr_model} models directory does not exist!')

    results_path = os.path.join(base_results_path, curr_model)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return models_path


def get_best_model_ep(models_path):
    '''
    This fuction gets the saved models path and returns the epoch of the best model (highest number in our current paradigm)
    '''
    # Find the best model
    best_epoch_model = 0
    for file in os.listdir(models_path):
        model_epoch = os.path.split(file)[1].split('.')[0].split('_')[-1]
        if int(model_epoch) >= best_epoch_model:
            best_epoch_model = int(model_epoch)

    return best_epoch_model


def kspace_to_image(tensor):
    '''
    This fuction turns a kspace tensor to an image.
    returns the image and the absolute shifted image
    '''
    # print(tensor.squeeze(0).shape)            # Debug
    _, n, m = tensor.squeeze(0).shape
    numpy_kspace = fastmri.tensor_to_complex_np(tensor.reshape((n, m, 2)))
    image = np.fft.fft2(numpy_kspace)
    return image, np.abs(np.fft.fftshift(image))


def inverse_mask(mask):
    '''
    This function gets a mask and returns its inverse mask
    '''
    zero_indices = mask == 0
    non_zero_indices = mask != 0
    mask[non_zero_indices] = 0
    mask[zero_indices] = 1
    return mask
