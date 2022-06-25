import torch
import config
from data_loader import loadFromDir, showKspaceFromTensor
import os
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from utils import init_paths, get_best_model_ep

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bt_size = config.TEST_BATCH_SIZE
# Define experiment path according to user and environment
if config.user == 'noga':
    data_base_path = config.NOGA_DATA_PATH
    exp_path = os.path.join(config.NOGA_EXP_PATH, config.EXP_NAME)
elif config.user == 'amitay':
    data_base_path = config.AMITAY_DATA_PATH
    exp_path = os.path.join(config.AMITAY_EXP_PATH, config.EXP_NAME)
elif config.user == 'triton':
    data_base_path = config.TRITON_DATA_PATH
    exp_path = os.path.join(config.TRITON_EXP_PATH, config.EXP_NAME)

base_models_path = os.path.join(exp_path, 'models')
if not os.path.exists(base_models_path):
    raise Exception('Error! Models directory does not exist!')

base_results_path = os.path.join(exp_path, 'results')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

re_model_st = 'real_values_model'
im_model_st = 'im_values_model'
re_models_path = init_paths(base_results_path, base_models_path, re_model_st)
im_models_path = init_paths(base_results_path, base_models_path, im_model_st)
# Find best models for both cnns
re_best_model = get_best_model_ep(re_models_path)
im_best_model = get_best_model_ep(im_models_path)
# Load best models to device
re_model = torch.load(os.path.join(re_models_path, f'model_ep_{re_best_model}.pth')).to(DEVICE)
im_model = torch.load(os.path.join(im_models_path, f'model_ep_{im_best_model}.pth')).to(DEVICE)

test_data = loadFromDir(data_base_path + 'val_data/', bt_size, 'test')

# TODO: put the correct losses
lossFunc = MSELoss()
# set models to evaluation mode
re_model.eval()
im_model.eval()
# turn off gradient tracking
with torch.no_grad():
    for (i, (y, x)) in enumerate(test_data):
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        re_prediction = re_model(x[:, 0, :, :])
        im_prediction = im_model(x[:, 1, :, :])
        pred = torch.stack([re_prediction, im_prediction], 1)
        print('MSE loss : {}'.format(lossFunc(pred, y)))
        for i_slice in range(0, 30, 5):
            plt.figure()
            showKspaceFromTensor(x[0, :, :, :])
            plt.suptitle('input')
            plt.figure()
            showKspaceFromTensor(pred[0, :, :, :])
            plt.suptitle('reconstruction result')
            plt.figure()
            showKspaceFromTensor(y[0, :, :, :])
            plt.suptitle('ground truth reconstruction')
            plt.show()

