import numpy as np
import random

def RandomMask(Kspace_slice, R, center_fraction):
    n,m = Kspace_slice.shape
    mask = np.zeros((n,m))
    num_center_lines = n*center_fraction
    mask[int(n/2-num_center_lines/2):int(n/2+num_center_lines/2),:] = 1
    sampled_inds = random.sample(range(0, n), int(n/R))
    mask[sampled_inds,:] = 1
    return mask, mask*Kspace_slice
