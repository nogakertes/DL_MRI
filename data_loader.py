import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
import fastmri
import matplotlib.pyplot as plt
import numpy as np

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8]
)

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    kspace = transforms.to_tensor(kspace)
    masked_kspace, _ = transforms.apply_mask(kspace, mask_func)
    return kspace, masked_kspace

dataset = mri_data.SliceDataset(
    # root=pathlib.Path('C:/Users/Nogas/Desktop/DL_MRI/train_data'),
    root=pathlib.Path('/Users/amitaylev/PycharmProjects/DL_MRI/train_data'),
    transform=data_transform,
    challenge='singlecoil')


for kspace,masked_kspace in dataset:
    plt.subplot(1,2,1)
    plt.imshow(np.log(fastmri.complex_abs(masked_kspace)+1e-9), cmap='gray')
    plt.subplot(1,2,2)
    image = fastmri.ifft2c(masked_kspace)
    plt.imshow(fastmri.complex_abs(image),cmap='gray')
    plt.show()



