# from Unet import *
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from models import *
from data_loader import loadFromDir, showKspaceFromTensor
from torch.nn import MSELoss
from torch.optim import Adam, SGD, RMSprop
import torch.nn.functional as F
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
import fastmri
import config
import utils


# Define experiment variables
NUM_EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
INIT_LR = config.LR
LR_PATIENCE = config.LR_PATIENCE
LR_FACTOR = config.LR_FACTOR

if config.CLEARML:
    from clearml import Task
    task = Task.init(project_name="DeepMRI", task_name=config.EXP_NAME)
    task.connect(config)
    logger = task.get_logger()

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

print('############################################################')
print(' ')
print('------------------BASE MODEL TRAINING------------------------')
print('pytorch is using {}'.format(DEVICE))

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

base_results_path = os.path.join(exp_path, 'results')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

if config.SAVE_NET:
    base_models_path = os.path.join(exp_path, 'models')
    if not os.path.exists(base_models_path):
        os.makedirs(base_models_path)

train_data = loadFromDir(data_base_path + 'train_data/', BATCH_SIZE, 'train')
val_data = loadFromDir(data_base_path + 'val_data/', BATCH_SIZE, 'val')

print('Number of training batches is {}'.format(len(train_data)))
print('Number of validation batches is {}'.format(len(val_data)))

model_re = Single_ch_U_Net().to(DEVICE)
model_im = Single_ch_U_Net().to(DEVICE)
# initialize loss function and optimizer
lossFunc = MSELoss()
# lossFunc = F.l1_loss()
optimizer_re = Adam(model_re.parameters(), lr=INIT_LR)
optimizer_im = Adam(model_im.parameters(), lr=INIT_LR)
# optimizer = SGD(model.parameters(), lr=INIT_LR)
# optimizer = RMSprop(model.parameters(), lr=INIT_LR)
lr = INIT_LR
# scheduler_re = ReduceLROnPlateau(optimizer_re, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
# scheduler_im = ReduceLROnPlateau(optimizer_im, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
scheduler_re = ExponentialLR(optimizer_re, gamma=LR_FACTOR)
scheduler_im = ExponentialLR(optimizer_im, gamma=LR_FACTOR)

# calculate steps per epoch for training and test set
trainSteps = len(train_data)
valSteps = len(val_data)

for i_model in range(2):
    if i_model == 0:
        curr_model = 'real_values_model'
        model = model_re
        optimizer = optimizer_re
        scheduler = scheduler_re
    else:
        curr_model = 'im_values_model'
        model = model_im
        optimizer = optimizer_im
        scheduler = scheduler_im

    # Set paths infrastructure
    results_path = os.path.join(base_results_path, curr_model)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if config.SAVE_NET:
        models_path = os.path.join(base_models_path, curr_model)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

    best_val_loss = 10000
    # initialize a dictionary to store training loss history
    H = {"train_loss": [], "val_loss": [], "lr": []}

    print('############################################################')
    print(' ')
    print(f'----------Started training {curr_model}----------')
    print(' ')
    print('############################################################')

    # Train the models sequentially
    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        ''' Training Loop '''
        for (i, (y, x)) in enumerate(train_data):
            # print(f'Training batch #{i}/{len(train_data)}')
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            x = x[:, i_model, :, :]
            y = y[:, i_model, :, :]
            # # Normalize x and y by min-max normalization
            # x = (x-x.min())/(x.max()-x.min())
            # y = (y-y.min())/(y.max()-y.min())
            pred = model(x)
            loss = lossFunc(pred, y.unsqueeze(1))     # for 1 input ch
            # zero previously accumulated gradients, then perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        print(f'Finished ep: {e} training with training loss: {totalTrainLoss}')

        ''' Validation Loop '''
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (i, (y, x)) in enumerate(val_data):
                # print(f'Validation batch #{i}/{len(val_data)}')
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                x = x[:, i_model, :, :]
                y = y[:, i_model, :, :]
                # make the predictions and calculate the validation loss
                pred = model(x)
                if config.DATA_CONSISTENCY:
                    consistency_x = x != 0
                    pred[consistency_x] = x[consistency_x]
                val_loss = lossFunc(pred, y.unsqueeze(1))       # for 1 input ch
                totalValLoss += val_loss
            print(f'Finished ep: {e} validation with validation loss: {totalValLoss}')

        ''' Calculations and scheduler step'''
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss)
        print(f'train_loss {avgTrainLoss}')
        H["val_loss"].append(avgValLoss)
        print(f'val_loss {avgValLoss}')
        H["lr"].append(lr)
        print(f'lr {lr}')

        # Save online plots per epoch on clearml - if defined
        if config.CLEARML:
            logger.report_scalar(title=curr_model+' Train and Validation Loss vs. Epochs', series='Train loss', value=avgTrainLoss, iteration=e)
            logger.report_scalar(title=curr_model+' Train and Validation Loss vs. Epochs', series='Validation loss', value=avgValLoss, iteration=e)
            logger.report_scalar(title=curr_model+' Learning Rate vs. Epochs', series='Learning Rate', value=lr, iteration=e)

        # print the model training and validation information
        print(curr_model + " EPOCH: {}/{}".format(e+1, NUM_EPOCHS))
        print(curr_model + " Train loss: {:.4f}, Validation loss: {:.4f}, Learning rate: {:.4f}".format(avgTrainLoss, avgValLoss, lr))
        # Save the best model so far
        if avgValLoss < best_val_loss and config.SAVE_NET:
            best_val_loss = avgValLoss
            utils.save_model(model, models_path=models_path, ep=e)

        # Decrease the lr by factor (new_lr = lr * factor) every #patientce epochs
        if e % LR_PATIENCE == 0 and e != 0:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            print(f'Defined new lr = {lr}')
        # Decrease the lr by factor (new_lr = lr * factor) if val_loss didn't improve over #patientce epochs
        # scheduler.step(avgValLoss)
        # lr = scheduler.get_last_lr()[0]
        # print(f'Defined new lr = {lr}')
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
    # plt.show()
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
    # plt.show()
    if config.SAVE_PLOTS:
        path = os.path.join(results_path, config.EXP_NAME + "_learning_rate_plot")
        plt.savefig(path)

    print('############################################################')
    print(' ')
    print(f'---------Finished training {curr_model}---------')
    print(' ')
    print('############################################################')
