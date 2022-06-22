import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import fastmri
import config

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.04],
    accelerations=[8]
)
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    """
    Transform the given k-space data by multiplying with a mask into the subsampled k-space tensor
    We first crop the center of the k-space image to specified height and width, then apply the mask on the cropped image

    """
    width, height = 320, 320
    kspace = transforms.to_tensor(kspace)
    kspace = transforms.complex_center_crop(kspace, shape=(width, height))
    #kspace = (kspace-kspace.min())/(kspace.max()-kspace.min()) #minmax normalization per sample
    kspace, _, _ = transforms.normalize_instance(kspace)    # add normalization to kspace
    if config.user == 'triton' or config.user == 'noga':
        masked_kspace, _, _ = transforms.apply_mask(kspace, mask_func)
    else:
        masked_kspace, _ = transforms.apply_mask(kspace, mask_func)
    return kspace.reshape((2, width, height)), masked_kspace.reshape((2, width, height))


def loadFromDir(dir_path, batch_size, data_type, remove_Edge_slices = None):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    dataset = mri_data.SliceDataset(
        root=pathlib.Path(dir_path),
        transform=data_transform,
        challenge='singlecoil')
    # Shuffle only the training data
    if data_type == 'train':
        shuffle_data = True
    else:
        shuffle_data = False
    loaded_data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)
    return loaded_data


def showKspaceFromTensor(tensor):
    _,n,m = tensor.shape
    numpy_kspace = fastmri.tensor_to_complex_np(tensor.reshape((n,m,2)))
    plt.subplot(1,2,1)
    plt.imshow(np.log(np.abs(numpy_kspace)+1e-9),cmap='gray')
    image = np.fft.fft2(numpy_kspace)
    plt.subplot(1,2,2)
    plt.imshow(np.abs(np.fft.fftshift(image)), cmap='gray')


