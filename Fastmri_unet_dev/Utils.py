'''
THis file is used for implement common used functions
'''
import numpy
import numpy as np
import matplotlib.pyplot as plt

import torch
import Data.Fastmri_Data_Processing
import Data.Data_Transform as dt
import torch.nn as nn
import torch.nn.functional as F
import h5py


def visualize_h5_slice(h5, slice_nums):
    '''
    Visualize the k-space, target

    Args:
        h5: a single H5Data(k-space, target, filename)
        slice_num: the specific slices we need to visualize

    '''

    ### Get volumn k-space
    filename = h5.file_name
    hf = h5py.File(filename)
    volume_kspace = hf['kspace'][()]                # shape (number of slices, height, width)

    ### visualize absolute value of k-space
    data = np.log(np.abs(volume_kspace) + 1e-9)
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=None)

def visualize_ifft_vs_target(ifft_img, target_img, slice_num):
    '''
    Visualize ifft image vs target image

    Args:
        ifft_img (numpy ndarray or torch tensor): image after applying Inverse Fourier Transform
        target_img (numpy ndarray): hdf5 dataset of volume (num_slices, width, height)
    '''

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(ifft_img[slice_num], cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(target_img[slice_num], cmap='gray')

def save_reconstruction(export_dir, reconstructions):
    '''
    Save reconstruction output
    Args:
        export_dir: save path for reconstruction output
        reconstructions: nd.array

    '''

    export_dir.mkdir(exist_ok=True, parents=True)
    for recons in reconstructions:
        with h5py.File(export_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])

