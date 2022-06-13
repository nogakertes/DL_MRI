import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data

from torch.utils.data import DataLoader

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8]
)
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    """
    Transform the given k-space data by multiplying with a mask into the subsampled k-space tensor
    We first crop the center of the k-space image to specified height and width, then apply the mask on the cropped image


    """
    width, height = 320, 320
    kspace = transforms.to_tensor(kspace)
    kspace = transforms.complex_center_crop(kspace, shape=(width, height))
    # # Plot the cropped image for debug
    # from matplotlib import pyplot as plt
    # import numpy as np
    # fig = plt.figure()
    # plt.imshow(np.log(np.abs(kspace[:, :, 0].numpy()) + 1e-9), cmap='gray')     #kspace is complex so choose real or im
    masked_kspace, _ = transforms.apply_mask(kspace, mask_func)
    # # Plot the masked kspace for debug
    # fig = plt.figure()
    # plt.imshow(np.log(np.abs(masked_kspace[:, :, 0].numpy()) + 1e-9), cmap='gray')
    return kspace.reshape((2, width, height)), masked_kspace.reshape((2, width, height))


def loadFromDir(dir_path, batch_size, data_type):
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
