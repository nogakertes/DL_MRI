# from Unet import *
from models import *
from data_loader import loadFromDir, showKspaceFromTensor
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
import fastmri
import config
from Unet import UNet


# Define experiment variables
NUM_EPOCHS = config.EPHOCHS
BATCH_SIZE = config.BATCH_SIZE
INIT_LR = config.LR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('############################################################')
print(' ')
print('------------------BASE MODEL TRAINING------------------------')
print('pytorch is using the {}'.format(DEVICE))

train_data = loadFromDir(config.TRAIN_DATA_PATH,BATCH_SIZE, 'train')
val_data = loadFromDir(config.VAL_DATA_PATH,BATCH_SIZE, 'val')
# train_data = loadFromDir('/Users/amitaylev/PycharmProjects/DL_MRI/train_data/', BATCH_SIZE, 'train')
# val_data = loadFromDir('/Users/amitaylev/PycharmProjects/DL_MRI/val_data/', BATCH_SIZE, 'val')
# train_data = loadFromDir('/home/stu1/singlecoil_train/', BATCH_SIZE, 'train')
# val_data = loadFromDir('/home/stu1/singlecoil_val/', BATCH_SIZE, 'val')

print('Number of training batches is {}'.format(len(train_data)))
print('Number of validation batches is {}'.format(len(val_data)))

#unet = UNet().to(DEVICE)
unet = U_Net().to(DEVICE)
# initialize loss function and optimizer
lossFunc = MSELoss()
opt = Adam(unet.parameters(), lr=INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(train_data) // BATCH_SIZE

if len(val_data) < BATCH_SIZE:
    valSteps = 1
else:
    valSteps = len(val_data) // BATCH_SIZE

# initialize a dictionary to store training loss history
H = {"train_loss": [], "val_loss": []}
for e in tqdm(range(NUM_EPOCHS)):
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    # loop over the training set
    for (i, (y, x)) in enumerate(train_data):
        # # Plot for debug before resize
        #fig = plt.figure()
        #showKspaceFromTensor(x[10, :, :, :])
        #fig = plt.figure()
        #showKspaceFromTensor(y[10, :, :, :])
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # # Plot for debug
        # fig = plt.figure()
        # showKspaceFromTensor(x[10, :, :, :])
        # fig = plt.figure()
        # showKspaceFromTensor(y[10, :, :, :])
        # plt.imshow(np.log(np.abs(x[10, 0, :, :].numpy()) + 1e-9), cmap='gray')
        # fig = plt.figure()
        # plt.imshow(np.log(np.abs(y[10, 0, :, :].numpy()) + 1e-9), cmap='gray')
        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = lossFunc(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss
    if e == NUM_EPOCHS-1:
        plt.figure()
        showKspaceFromTensor(x[5, :, :, :].detach())
        plt.suptitle('input-real value')
        plt.figure()
        showKspaceFromTensor(pred[5, :, :, :].detach())
        plt.suptitle('reconstruction result-real value')
        plt.figure()
        showKspaceFromTensor(y[5, :, :, :].detach())
        plt.suptitle('ground truth reconstruction-real value')
        plt.show()

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (y,x) in val_data:
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            totalTestLoss += lossFunc(pred, y)
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / valSteps
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgTestLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
        avgTrainLoss, avgTestLoss))


# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()
#plt.savefig(config.PLOT_PATH)
# serialize the model to disk
if config.SAVE_NET:
    path = os.path.join(config.PATH_TO_SAVE_NET, "baseModel.pth")
    torch.save(unet, path)