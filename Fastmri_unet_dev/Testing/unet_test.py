import torch
import torch.nn
import numpy as np
import pickle
import config_file
import matplotlib.pyplot as plt

import fastmri
from fastmri.data import transforms as T

from collections import defaultdict



def test(args, model, test_loader):
    '''

    Args:
        model: pretrained model used to reconstruct MR image
        test_loader: test data loader

    Returns:
        reconstruction: Dictionary of mapping from file name to reconstruction image
    '''
    # loading the model
    model = pickle.load(open(model, 'rb'))
    model.to(args.device)
    model.eval()
    reconstructions = []
    input_list = []
    target_list = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.to(device=args.device)
            inputs = complex_to_magnitude(inputs)
            inputs = inputs.float()

            targets = targets.to(device=args.device)
            targets = complex_to_magnitude(targets)
            targets = targets.float()

            pred = model.forward(inputs)

            N = len(inputs)                     # batch size N
            # num_channel = len(config_file.SLICES)    # number of channel
            num_channel = 3
            for i in range(N):
                for j in range(num_channel):
                    reconstructions.append(pred[i][j].cpu().numpy())        # torch tensor needs to be converted to nd.array
                    input_list.append(inputs[i][j].cpu().numpy())
                    target_list.append(targets[i][j].cpu().numpy())

    # reconstructions = np.stack(reconstructions, axis=0)

    slice_num = 20

    fig = plt.figure()
    plt.imshow(input_list[slice_num], cmap='gray')
    plt.savefig('slice1_dp0.0_win11_input.png')

    plt.imshow(reconstructions[slice_num], cmap='gray')
    plt.savefig('slice1_dp0.0_win11_recon.png')

    plt.imshow(target_list[slice_num], cmap='gray')
    plt.savefig('slice1_dp0.0_win11_target.png')
    # plt.show()


# print(recon)

    # return recon, targets


def complex_to_magnitude(data):
    # Ensure the input is complex and has an imaginary part
    if torch.is_complex(data):
        magnitude = torch.abs(data)
    else:
        magnitude = data  # Assuming data is already real
    return magnitude