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
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    width, height = 350,350
    kspace = transforms.to_tensor(kspace)
    kspace = transforms.complex_center_crop(kspace,shape=(width, height))
    masked_kspace, _ = transforms.apply_mask(kspace, mask_func)
    return kspace.reshape((2,width, height)), masked_kspace.reshape((2,width, height))


def loadFromDir(dir_path,BATCH_SIZE):
    dataset = mri_data.SliceDataset(
        root=pathlib.Path(dir_path),
        transform=data_transform,
        challenge='singlecoil')
    train_data = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle = True)
    return train_data
