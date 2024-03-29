# from Unet import *
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from models import *
from data_loader import loadFromDir, showKspaceFromTensor
from torch.nn import MSELoss,L1Loss
from torch.optim import Adam, SGD, RMSprop
import torch.nn.functional as F
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
import fastmri
import config
import utils
from losses import SSIMLoss


# Define experiment parameters
NUM_EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
INIT_LR = config.LR
LR_PATIENCE = config.LR_PATIENCE
LR_FACTOR = config.LR_FACTOR

# Open a clearml task if defined
if config.CLEARML:
    from clearml import Task
    task = Task.init(project_name="DeepMRI", task_name=config.EXP_NAME)
    task.connect(config)
    logger = task.get_logger()

# Choose the best device possible
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

print('############################################################')
print(' ')
print('------------------BASE MODEL TRAINING------------------------')
print('pytorch is using the {}'.format(DEVICE))

# Define experiment paths according to user and environment
if config.user == 'noga':
    data_base_path = config.NOGA_DATA_PATH
    exp_path = os.path.join(config.NOGA_EXP_PATH, config.EXP_NAME)
elif config.user == 'amitay':
    data_base_path = config.AMITAY_DATA_PATH
    exp_path = os.path.join(config.AMITAY_EXP_PATH, config.EXP_NAME)
elif config.user == 'triton':
    data_base_path = config.TRITON_DATA_PATH
    exp_path = os.path.join(config.TRITON_EXP_PATH, config.EXP_NAME)

# Define specific paths for experiment uotputs
results_path = os.path.join(exp_path, 'results')
if not os.path.exists(results_path):
    os.makedirs(results_path)

if config.SAVE_NET:
    models_path = os.path.join(exp_path, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

# Get the train and val dataloaders
train_data = loadFromDir(data_base_path + 'train_data/', BATCH_SIZE, 'train')
val_data = loadFromDir(data_base_path + 'val_data/', BATCH_SIZE, 'val')
print('Number of training batches is {}'.format(len(train_data)))
print('Number of validation batches is {}'.format(len(val_data)))

# Initialize the model and attach to device
model = U_Net().to(DEVICE)
# initialize loss function and optimizer
lossFunc = MSELoss()
# lossFunc = L1Loss()
optimizer = Adam(model.parameters(), lr=INIT_LR)
# optimizer = SGD(model.parameters(), lr=INIT_LR)
# optimizer = RMSprop(model.parameters(), lr=INIT_LR)
lr = INIT_LR
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
scheduler = ExponentialLR(optimizer, gamma=LR_FACTOR)
reg_loss = SSIMLoss()
reg_factor = 0.01
# calculate steps per epoch for training and test set
trainSteps = len(train_data)
valSteps = len(val_data)

# Define a super high val loss to find the best validation losses during training
best_val_loss = 10000
# initialize a dictionary to store training loss history
H = {"train_loss": [], "val_loss": [], "lr": []}

# Start model training
for e in tqdm(range(NUM_EPOCHS)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    ''' Training Loop '''
    for (i, (y, x)) in enumerate(train_data):
        # perform a forward pass and calculate the training loss
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # Forward pass in the model
        pred = model(x)
        # Adding data consistency if defined
        if config.DATA_CONSISTENCY:
            consistency_x = x != 0
            pred[consistency_x] = x[consistency_x]
        # Calculation of the loss
        loss = lossFunc(pred, y)
        # zero previously accumulated gradients, then perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss
    # Plot several epoch results during training
    if e % 10 == 0 or e == NUM_EPOCHS-1:
        plt.figure()
        showKspaceFromTensor(x[5, :, :, :].cpu().detach())
        plt.suptitle(f'Epoch #{e} training downsampled input')
        path = os.path.join(results_path, config.EXP_NAME + f'_epoch_#{e}_downsampled_k-space_input_image')
        plt.savefig(path)
        showKspaceFromTensor(pred[5, :, :, :].cpu().detach())
        plt.suptitle(f'Epoch #{e} training reconstruction output')
        path = os.path.join(results_path, config.EXP_NAME + f'_epoch_#{e}_reconstructed_k-space_image')
        plt.savefig(path)
        plt.figure()
        showKspaceFromTensor(y[5, :, :, :].cpu().detach())
        plt.suptitle(f'Epoch #{e} training ground truth')
        path = os.path.join(results_path, config.EXP_NAME + f'_epoch_#{e}_GT_k-space_image')
        plt.savefig(path)

    ''' Validation Loop '''
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (y, x) in val_data:
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            # Add the data consistency if defined
            if config.DATA_CONSISTENCY:
                consistency_x = x != 0
                pred[consistency_x] = x[consistency_x]
            # Calculate validation loss
            val_loss = lossFunc(pred, y)
            totalValLoss += val_loss

    ''' Calculations and scheduler step'''
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["lr"].append(lr)

    # Save online plots per epoch on clearml - if defined
    if config.CLEARML:
        logger.report_scalar(title='Train and Validation Loss vs. Epochs', series='Train loss', value=avgTrainLoss, iteration=e)
        logger.report_scalar(title='Train and Validation Loss vs. Epochs', series='Validation loss', value=avgValLoss, iteration=e)
        logger.report_scalar(title='Learning Rate vs. Epochs', series='Learning Rate', value=lr, iteration=e)

    # print the model training and validation information
    print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Validation loss: {:.4f}, Learning rate: {:.10f}".format(
        avgTrainLoss, avgValLoss, lr))
    # Save the best model so far
    if avgValLoss < best_val_loss and config.SAVE_NET:
        best_val_loss = avgValLoss
        # print(f'Best model so far is saved from epoch: {e}')
        utils.save_model(model, models_path=models_path, ep=e)

    # Decrease the lr by factor (new_lr = lr * factor) every #patientce epochs
    if e % LR_PATIENCE == 0 and e != 0:
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f'Defined new lr = {lr}')
    # # Optional use of Reduce on plateau scheduler
    # Decrease the lr by factor (new_lr = lr * factor) if val_loss didn't improve over #patientce epochs
    # scheduler.step(avgValLoss)
    # lr = scheduler.get_last_lr()[0]
    # print(f'Defined new lr = {lr}')

    # Stop the training process if the lr is too small
    if lr < 1e-7:
        print(f'lr is {lr} and is smaller than 1e-7. Stopping the train!')

''' Plots '''
# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Train and Validation Loss vs. Epochs")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()

if config.SAVE_PLOTS:
    path = os.path.join(results_path, config.EXP_NAME + "_train_val_loss_plot")
    plt.savefig(path)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["lr"], label="learning_rate")
plt.title("Learning Rate vs. Epochs")
plt.xlabel("Epoch #")
plt.ylabel("Learning Rate")
plt.legend(loc="lower left")
plt.show()

if config.SAVE_PLOTS:
    path = os.path.join(results_path, config.EXP_NAME + "_learning_rate_plot")
    plt.savefig(path)
