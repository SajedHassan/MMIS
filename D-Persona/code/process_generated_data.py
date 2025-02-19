import torch
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import nibabel as nib
from configs.config import *
from utils.utils import rand_seed, show_img
from lib.metrics_set import *
from dataloader.dataset import BaseDataSets, ZoomGenerator
from torch.utils.data import DataLoader
from lib.initialize_model_single_path import init_model
import h5py
import re
from scipy.ndimage.interpolation import zoom


if __name__ == '__main__':
    reference_folder_path = '/home/sajed/thesis/MMIS/dataset/ORG_1_ONLY/training_2d/'
    
    # Loop over all .nii.gz generated masks
    sorted_files_list = sorted(os.listdir(reference_folder_path), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])
    index = 0
    for file_name in sorted_files_list:
        # Read data from HDF5 file
        match = re.search(r"Sample_(\d+)_", file_name)
        annotator = int(int(match.group(1)) / 25)
        output_path = '/home/sajed/thesis/MMIS/dataset/ORG_1_ONLY_stage2_GENERATED/training_2d/'
        h5file = h5py.File(reference_folder_path + file_name, "r")
        annotator_mask = np.array(h5file["label_a" + str(annotator + 1)][:])
        not_processed_folder = '/home/sajed/thesis/MMIS/dataset/ORG_1_ONLY_stage2_GENERATED_NOT_PROCESSED/'
        image_index = str(index)
        if index < 10:
            image_index = '0' + image_index
        not__processed_pred1 = annotator_mask if annotator == 0 else nib.load(not_processed_folder + image_index + '_pred_s1.nii.gz').get_fdata()[0][0]
        not__processed_pred2 = annotator_mask if annotator == 1 else nib.load(not_processed_folder + image_index + '_pred_s2.nii.gz').get_fdata()[0][0]
        not__processed_pred3 = annotator_mask if annotator == 2 else nib.load(not_processed_folder + image_index + '_pred_s3.nii.gz').get_fdata()[0][0]
        not__processed_pred4 = annotator_mask if annotator == 3 else nib.load(not_processed_folder + image_index + '_pred_s4.nii.gz').get_fdata()[0][0]

        output_size = annotator_mask.shape
        not__processed_pred1 = zoom(not__processed_pred1, (output_size[0] / not__processed_pred1.shape[0], output_size[1] / not__processed_pred1.shape[1]), order=0)
        not__processed_pred2 = zoom(not__processed_pred2, (output_size[0] / not__processed_pred2.shape[0], output_size[1] / not__processed_pred2.shape[1]), order=0)
        not__processed_pred3 = zoom(not__processed_pred3, (output_size[0] / not__processed_pred3.shape[0], output_size[1] / not__processed_pred3.shape[1]), order=0)
        not__processed_pred4 = zoom(not__processed_pred4, (output_size[0] / not__processed_pred4.shape[0], output_size[1] / not__processed_pred4.shape[1]), order=0)

        stacked_image = np.hstack((not__processed_pred1, not__processed_pred2, not__processed_pred3, not__processed_pred4))
        

        with h5py.File(output_path + file_name, "w") as output:  # Use "w" to create or overwrite the file
            output.create_dataset("t1", data=np.array(h5file['t1']))
            output.create_dataset("t1c", data=np.array(h5file['t1c']))
            output.create_dataset("t2", data=np.array(h5file['t2']))

            output.create_dataset("label_a1", data=not__processed_pred1)
            output.create_dataset("label_a2", data=not__processed_pred2)
            output.create_dataset("label_a3", data=not__processed_pred3)
            output.create_dataset("label_a4", data=not__processed_pred4)
        
        h5file.close()
        index += 1
