import torch
import config
from data_loader import loadFromDir, showKspaceFromTensor
import os
import matplotlib.pyplot as plt
from torch.nn import MSELoss
import random
from losses import ssim

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
# if not os.path.exists(models_path):
#     raise('Error! Models directory does not exist!')

test_data = loadFromDir(data_base_path + 'test_data/', bt_size, 'test')
lossFunc = MSELoss()

# Find the best model
best_epoch_model = 0
for file in os.listdir(models_path):
    model_epoch = os.path.split(file)[1].split('.')[0].split('_')[-1]
    if int(model_epoch) >= best_epoch_model:
        best_epoch_model = int(model_epoch)

# Run inference using the best model
model_path = os.path.join(models_path, f'model_ep_{best_epoch_model}.pth')
model = torch.load(model_path).to(DEVICE)
results_path = os.path.join(exp_path, 'results')
total_ssim_score = 0
# set model to evaluation mode
model.eval()
# turn off gradient tracking
with torch.no_grad():
    samples = random.sample(range(0,len(test_data)),5)
    for (i, (y, x)) in enumerate(test_data):
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        pred = model(x)
        total_ssim_score += ssim(y.cpu().detach(),pred.cpu().detach())
        if i in samples:
            plt.figure()
            showKspaceFromTensor(x[0, :, :, :].cpu().detach())
            plt.suptitle('input_test_sample_{}'.format(i))
            plt.figure()
            path = os.path.join(results_path, config.EXP_NAME + '_input_test_{}'.format(i))
            plt.savefig(path)
            showKspaceFromTensor(pred[0, :, :, :].cpu().detach())
            plt.suptitle('reconstruction result')
            path = os.path.join(results_path, config.EXP_NAME + '_reconstruction result_test_{}'.format(i))
            plt.savefig(path)
            plt.figure()
            showKspaceFromTensor(y[0, :, :, :].cpu().detach())
            plt.suptitle('ground truth reconstruction_{}'.format(i))
            path = os.path.join(results_path, config.EXP_NAME + '_ground truth reconstruction_test_{}'.format(i))
            plt.savefig(path)
            plt.show()

print('mean ssim score is : {}'.format(total_ssim_score/len(test_data)))
