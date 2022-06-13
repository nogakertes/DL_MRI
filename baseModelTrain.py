from Unet import *
from data_loader import loadFromDir
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np


# Define experiment variables
NUM_EPOCHS = 10
BATCH_SIZE = 16
INIT_LR = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('pytorch is using the {}'.format(DEVICE))

# train_data = loadFromDir('C:/Users/Nogas/Desktop/fastmri_data/train_data/',BATCH_SIZE, 'train')
# val_data = loadFromDir('C:/Users/Nogas/Desktop/fastmri_data/val_data/',BATCH_SIZE, 'val')
train_data = loadFromDir('/Users/amitaylev/PycharmProjects/DL_MRI/train_data/', BATCH_SIZE, 'train')
val_data = loadFromDir('/Users/amitaylev/PycharmProjects/DL_MRI/val_data/', BATCH_SIZE, 'val')

print('Number of training batches is {}'.format(len(train_data)))
print('Number of validation batches is {}'.format(len(val_data)))

unet = UNet().to(DEVICE)
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
        # Resize the images to be 128x128
        x = F.interpolate(x, size=128)
        y = F.interpolate(y, size=128)
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # # Plot for debug
        # fig = plt.figure()
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
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (x, y) in val_data:
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
path = os.path.join('/home/stu1', "unet_tgs_salt.pth")
torch.save(unet, path)