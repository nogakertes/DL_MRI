import numpy as np
import torch
import config
from data_loader import loadFromDir, showKspaceFromTensor
import os
import matplotlib.pyplot as plt
from torch.nn import MSELoss
import random
from losses import ssim, calculate_ssim, py_ssim, calculate_psnr
from utils import get_best_model_ep, kspace_to_image, inverse_mask

# Choose the best device possible

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize and define experiment parameters and paths according to config file
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

# Define outputs paths
models_path = os.path.join(exp_path, 'models')
if not os.path.exists(models_path):
    raise Exception('Error! Models directory does not exist!')

results_path = os.path.join(exp_path, 'evaluation_results')
if not os.path.exists(results_path):
    os.makedirs(results_path)

accuracy_file_path = os.path.join(results_path, 'accuracy_results.txt')

# Get the test set dataloader
test_data = loadFromDir(data_base_path + 'test_data/', bt_size, 'test')

# Find the best model to load (i.e latest model for current training paradigm)
best_epoch_model = get_best_model_ep(models_path)

# Load the best model and define a loss function for accuracy score calculations
inference_model_path = os.path.join(models_path, f'model_ep_{best_epoch_model}.pth')
model = torch.load(inference_model_path).to(DEVICE)
lossFunc = MSELoss()

# Initialize accuracy calculation parameters
total_ssim_score = 0
total_ssim_score_image_shift = 0
total_mse_score = 0
total_psnr_shift_score = 0

# set model to evaluation mode
model.eval()
# Run inference using the best model (turn off gradient tracking)
with torch.no_grad():
    # Define a random test sampling mechanism
    samples = random.sample(range(0, len(test_data)), 10)
    # Evaluate the model
    for (i, (y, x)) in enumerate(test_data):
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # Run a forward pass on the model
        pred = model(x)
        # Add data consistency if defined
        if config.DATA_CONSISTENCY:
            consistency_x = x != 0
            pred[consistency_x] = x[consistency_x]
        # Calculate metrics
        # MSE loss on kspace
        curr_MSE = lossFunc(pred, y)
        total_mse_score += curr_MSE
        # Get images out of kspace for predictions and GT
        pred_image_space, pred_image_space_shift = kspace_to_image(pred.cpu().detach())
        y_image_space, y_image_space_shift = kspace_to_image(y.cpu().detach())
        # PSNR calculations on image domain
        curr_psnr_shift = calculate_psnr(y_image_space_shift, pred_image_space_shift,y_image_space_shift.max())
        total_psnr_shift_score += curr_psnr_shift
        # SSIM calculations on image domain
        curr_ssim_imgae_shift = calculate_ssim(y_image_space_shift, pred_image_space_shift,y_image_space_shift.max())
        total_ssim_score_image_shift += curr_ssim_imgae_shift
        # SSIM calculations on kspace
        curr_ssim = ssim(y.cpu().detach(), pred.cpu().detach())
        total_ssim_score += curr_ssim

        # Sanity prints every 100 epochs
        if i % 100 == 0:
            print(f'Test sample {i}, image_PSNR: {curr_psnr_shift:.4f}, kspace_SSIM: {curr_ssim:.4f}, SSIM_image_shift: {curr_ssim_imgae_shift:.4f}, MSE: {curr_MSE:.4f}')

        # Plot and print few samples accorrdin to what randomly defined above
        if i in samples:
            # Input
            plt.figure()
            showKspaceFromTensor(x[0, :, :, :].cpu().detach())
            plt.suptitle(f'Test input {i}, kspace SSIM: {curr_ssim:.4f}, image SSIM: {curr_ssim_imgae_shift:.4f}, MSE: {curr_MSE:.4f}')
            path = os.path.join(results_path, config.EXP_NAME + f'_test_input_{i}')
            # plt.show()
            plt.savefig(path)
            plt.close()

            # Prediction
            plt.figure()
            showKspaceFromTensor(pred[0, :, :, :].cpu().detach())
            plt.suptitle(f'Test prediction {i}, kspace SSIM: {curr_ssim:.4f}, image SSIM: {curr_ssim_imgae_shift:.4f}, MSE: {curr_MSE:.4f}')
            path = os.path.join(results_path, config.EXP_NAME + f'_test_reconstruction_output_{i}')
            # plt.show()
            plt.savefig(path)
            plt.close()

            # Ground truth
            plt.figure()
            showKspaceFromTensor(y[0, :, :, :].cpu().detach())
            plt.suptitle(f'GT sample {i}, kspace SSIM: {curr_ssim:.4f}, image SSIM: {curr_ssim_imgae_shift:.4f}, MSE: {curr_MSE:.4f}')
            path = os.path.join(results_path, config.EXP_NAME + f'_test_GT_{i}')
            # plt.show()
            plt.savefig(path)
            plt.close()

# Print accuracy results and save them in a txt file
with open(accuracy_file_path, "w") as f:
    f.write(f'Experiment {config.EXP_NAME} results:')
    f.write('\nmean SSIM score is : {:.4f}'.format(total_ssim_score/len(test_data)))
    f.write('\nmean image PSNR score is : {:.4f}'.format(total_psnr_shift_score/len(test_data)))
    f.write('\nmean image SSIM score is : {:.4f}'.format(total_ssim_score_image_shift/len(test_data)))
    f.write('\nmean MSE score is : {:.4f}'.format(total_mse_score/len(test_data)))

print('###########################################################')
print(' ')
print('-------------- Finished evaluation --------------')
print(f'Experiment {config.EXP_NAME} results:')
print('mean SSIM score is : {:.4f}'.format(total_ssim_score/len(test_data)))
# print('mean PSNR score is : {:.4f}'.format(total_psnr_score/len(test_data)))
print('mean image PSNR score is : {:.4f}'.format(total_psnr_shift_score/len(test_data)))
# print('mean SSIM image score is : {:.4f}'.format(total_ssim_score_image/len(test_data)))
print('mean image SSIM score is : {:.4f}'.format(total_ssim_score_image_shift/len(test_data)))
print('mean MSE score is : {:.4f}'.format(total_mse_score/len(test_data)))
