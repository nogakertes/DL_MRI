import h5py
import numpy as np
from matplotlib import pyplot as plt
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms as T
import fastmri


# file_name = 'multicoil_train/file1000167.h5'
file_name = '/Users/amitaylev/PycharmProjects/DL_MRI/train_data/file1000592.h5'
hf = h5py.File(file_name)
print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))
volume_kspace = hf['kspace'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)
slice_kspace = volume_kspace[20] # Choosing the 20-th slice of this volume


def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)


show_coils(np.log(np.abs(volume_kspace) + 1e-9), [0, 5, 10])  # This shows coils 0, 5 and 10

volume_kspace2 = T.to_tensor(volume_kspace)      # Convert from numpy array to pytorch tensor
volume_image = fastmri.ifft2c(volume_kspace2)
volume_image_abs = fastmri.complex_abs(volume_image)         # Apply Inverse Fourier Transform to get the complex image

show_coils(volume_image_abs, [0, 5, 10], cmap='gray')   # Compute absolute value to get a real image

volume_image_rss = fastmri.rss(volume_image_abs, dim=0)     # get the full image from all the coils

fig2 = plt.figure()
plt.imshow(np.abs(volume_image_rss.numpy()), cmap='gray')

mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object
masked_kspace, mask = T.apply_mask(volume_kspace2, mask_func)   # Apply the mask to k-space

sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')

print('Done')
