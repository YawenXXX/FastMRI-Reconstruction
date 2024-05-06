"""
This code is for running the whole process
"""
import glob, os

import numpy as np
import torch

import argparse

from torch.utils.data import DataLoader

import config_file

# import Model.unet as Unet
from fastmri.models import Unet
import Data.Fastmri_Data_Processing as dp
import Training.losses as losses
import Training.unet_train as unet_train
import Testing.unet_test as unet_test

import Utils


ROOT_DIR = os.path.abspath('.')

def arg_parser():
    parser = argparse.ArgumentParser(description='Train Unet for MRI Reconstruction'
                                     , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', default=config_file.BATCH_SIZE, type=int, help='Batch Size')
    parser.add_argument('--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--device', type=str, default='cuda', help='device to run on')
    parser.add_argument('--seed', type=int, default=430, help='Fdix random seed')

    args = parser.parse_args()
    return args

def main(args, mode='train'):

    input_data_dir = config_file.INPUT_DATA_DIR
    input_validation_dir = config_file.INPUT_VALID_DATA_DIR
    input_annotation_dir = config_file.INPUT_ANOTATION_DIR
    input_test_dir = config_file.INPUT_TEST_DIR
    reconstruction_dir = config_file.RECONSTRUCTION_DIR

    slice_idxs = config_file.SLICES
    crop_size = config_file.CROP_SIZE
    batch_size = config_file.BATCH_SIZE
    max_num_data = config_file.MAX_FILE_LIMIT

    h5_lists = dp.get_h5_file_list(input_data_dir, max_num_data)
    h5_validate_lists = dp.get_h5_file_list(input_validation_dir, max_num_data)
    h5_test_lists = dp.get_h5_file_list(input_test_dir, max_num_data)[:10]

    if input_annotation_dir != '':
        annotation_lists = dp.get_annotation_file_list(input_annotation_dir)

    if len(h5_lists) == 0:
        print("There is no h5 file input.")
        return
    # Create dataset
    fastmri_dataset = dp.FastMriDataset(h5_lists, slice_idxs, crop_size)
    fastmri_validate_dataset = dp.FastMriDataset(h5_validate_lists, slice_idxs, crop_size)
    fastmri_test_dataset = dp.FastMriDataset(h5_test_lists, slice_idxs, crop_size)

    # Create data loader
    train_loader = DataLoader(fastmri_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(fastmri_validate_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(fastmri_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_channel = train_loader.dataset[0][0].shape[0]
    num_channel_first_layer_output = config_file.NUM_CHANNEL_FIRST_LAYER_OUTPUT
    num_pool_layers = config_file.NUM_POOL_LAYERS
    dropout_prob = config_file.DROPOUT_PROB

    # print("num_channel: ", num_channel)
    # print("num_channel_first_layer_output: ", num_channel_first_layer_output)
    # print("num_pool_layers: ", num_pool_layers)
    # print("dropout_prob: ", dropout_prob)

    model = Unet(in_chans=num_channel,
                 out_chans=num_channel,
                 chans=num_channel_first_layer_output,
                 num_pool_layers=num_pool_layers,
                 drop_prob=dropout_prob)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if mode == 'train':
        # Training
        val_loss_log, best_val_loss = unet_train.train(args, model, True, optimizer, train_loader, validation_loader)
        print(best_val_loss)

    elif mode == 'test':
        # Output and save reconstructions
        saved_model_path = config_file.OUTPUT_DIR
        saved_models = sorted(glob.glob(os.path.join(saved_model_path, '*.pkl')), key=os.path.getmtime)
        best_model = saved_models[38]

        reconstructions = unet_test.test(args, best_model, test_loader)
        # Utils.save_reconstruction(reconstruction_dir, reconstructions)

if __name__ == '__main__':
    args = arg_parser()

    main(args, mode='train')




