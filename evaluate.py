import torch
import config
from data_loader import loadFromDir, showKspaceFromTensor
import os
import matplotlib.pyplot as plt
from torch.nn import MSELoss


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

models_path = os.path.join(exp_path, 'models')
if not os.path.exists(models_path):
    raise 'Error! Models directory does not exist!'

test_data = loadFromDir(data_base_path + 'val_data/', bt_size, 'test')
lossFunc = MSELoss()


def make_predictions(model, data):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        for (i, (y, x)) in enumerate(data):
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            pred = model(x)
            print('MSE loss : {}'.format(lossFunc(pred, y)))
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


# Find the best model
best_epoch_model = 0
for file in os.listdir(models_path):
    model_epoch = os.path.split(file)[1].split('.')
    if model_epoch > best_epoch_model:
        best_epoch_model = model_epoch

# Run inference using the best model
model_path = os.path.join(models_path, f'model_ep_{best_epoch_model}.pth')
model = torch.load(model_path).to(DEVICE)
make_predictions(model, test_data)
