import torch
import config
from data_loader import loadFromDir, showKspaceFromTensor
import os
import matplotlib.pyplot as plt
from torch.nn import MSELoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_data = loadFromDir(config. VAL_DATA_PATH,1, 'test')
lossFunc = MSELoss()
def make_predictions(model,data):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        for (i, (y, x)) in enumerate(data):
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            pred = unet(x)
            print('MSE loss : {}'.format(lossFunc(pred, y)))
            plt.figure()
            showKspaceFromTensor(x[0, :, :, :])
            plt.suptitle('input')
            plt.figure()
            showKspaceFromTensor(pred[0,:,:,:])
            plt.suptitle('reconstruction result')
            plt.figure()
            showKspaceFromTensor(y[0,:,:,:])
            plt.suptitle('ground truth reconstruction')
            plt.show()


net_path = os.path.join(config.PATH_TO_SAVE_NET,'baseModel.pth')
unet = torch.load(net_path).to(DEVICE)
make_predictions(unet,test_data)
