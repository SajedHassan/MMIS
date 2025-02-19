import os
import sys
import h5py
import numpy as np
import nibabel as nib
DPersona_path = os.path.abspath("D-Persona/code")
sys.path.append(DPersona_path)
from lib.metrics_set import *
import torch
from collections import defaultdict

original_dataset_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/testing/'
generated_dataset_dir = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_output/Dataset022_NPC-3d_learnable_emb_spade_enc_dec_with_validation_testing_data_PP/'

all_samples_names = sorted(os.listdir(original_dataset_dir), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])

print(len(all_samples_names))

sample_index = 1

GED_global, Dice_max, Dice_max_reverse, Dice_soft, Dice_match, Dice_each = 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0] * 4)

for idx, sample_name in enumerate(all_samples_names):
    print(f"{sample_name}")

    original_sample = h5py.File(os.path.join(original_dataset_dir, sample_name), 'r')

    original_label_a1 = np.array(original_sample['label_a1'])
    original_label_a2 = np.array(original_sample['label_a2'])
    original_label_a3 = np.array(original_sample['label_a3'])
    original_label_a4 = np.array(original_sample['label_a4'])

    masks = np.stack([original_label_a1, original_label_a2, original_label_a3, original_label_a4])
    sample_masks = masks
    preds = []

    original_slice_name_parts = sample_name.replace('.', '_').split("_")
    for generated_AnnotatorIdx in range(0, 4):
        
        new_sample_base_path = 'NPC' + original_slice_name_parts[1] + '-' + str(generated_AnnotatorIdx) + '_' + "{:03}".format(sample_index)

        pred = nib.load(os.path.join(generated_dataset_dir, new_sample_base_path) + '.nii.gz').get_fdata()
        preds.append(pred)
    
    preds = np.stack(preds)
    sample_preds = preds
    sample_index += 1

    sample_masks = torch.tensor([np.stack(sample_masks, 0)]).float()
    sample_preds = torch.tensor([np.stack(sample_preds, 0)]).float()

    GED_iter = generalized_energy_distance(sample_masks, sample_preds)
    # Dice score
    dice_max_iter, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(sample_masks, sample_preds, thresh=0.5)
    dice_soft_iter = dice_at_thresh(sample_masks, sample_preds)

    GED_global += GED_iter
    Dice_match += dice_match_iter
    Dice_max += dice_max_iter
    Dice_max_reverse += dice_max_reverse_iter
    Dice_soft += dice_soft_iter
    Dice_each += np.array(dice_each_iter)

metrics_dict = {'GED': GED_global / len(all_samples_names),
                    'Dice_max': Dice_max / len(all_samples_names),
                    'Dice_max_reverse': Dice_max_reverse / len(all_samples_names),
                    'Dice_max_mean': (Dice_max_reverse + Dice_max) / (2 * len(all_samples_names)),
                    'Dice_match': Dice_match / len(all_samples_names),
                    'Dice_soft': Dice_soft / len(all_samples_names),
                    'Dice_each': Dice_each / len(all_samples_names),
                    'Dice_each_mean': np.mean(Dice_each) / len(all_samples_names)}

for key in metrics_dict.keys():
    print(key, ': ', metrics_dict[key])