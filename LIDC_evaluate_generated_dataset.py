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

original_dataset_dir = '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/val/'
generated_dataset_dir = '/home/sajed/thesis/nnUNet/nnunetv2/nnUNet_output/Dataset024_LIDC-learnable_emb_spade_enc_dec_FOLD_1_validation_data/'

all_slices_names = sorted(os.listdir(original_dataset_dir), key=lambda s: [int(part) if part.isdigit() else part for part in s.replace('.', '_').split("_")])

print(len(all_slices_names))

sample_index = 1

GED_global, Dice_max, Dice_max_reverse, Dice_soft, Dice_match, Dice_each = 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0] * 4)

for slice_name in all_slices_names:
    original_slice = h5py.File(os.path.join(original_dataset_dir, slice_name), 'r')

    original_label_a1 = np.array(original_slice['label_a1'])
    original_label_a2 = np.array(original_slice['label_a2'])
    original_label_a3 = np.array(original_slice['label_a3'])
    original_label_a4 = np.array(original_slice['label_a4'])

    masks = np.stack([original_label_a1, original_label_a2, original_label_a3, original_label_a4])

    preds = []
    original_slice_name_parts = slice_name.replace('.', '_').split("_")
    for generated_AnnotatorIdx in range(0, 4):
        
        new_sample_base_path = 'LIDC-' + original_slice_name_parts[0] + '-' + original_slice_name_parts[1] + '-' + str(generated_AnnotatorIdx) + '_' + "{:03}".format(sample_index)

        pred = nib.load(os.path.join(generated_dataset_dir, new_sample_base_path) + '.nii.gz').get_fdata()
        preds.append(pred)
    
    preds = np.stack(preds)

    masks = torch.tensor([np.stack(masks, 0)]).float()
    preds = torch.tensor([np.stack(preds, 0)]).float()

    GED_iter = generalized_energy_distance(masks, preds)
    dice_max_iter, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(masks, preds, thresh=0.5)
    dice_soft_iter = dice_at_thresh(masks, preds)

    GED_global += GED_iter
    Dice_match += dice_match_iter
    Dice_max += dice_max_iter
    Dice_max_reverse += dice_max_reverse_iter
    Dice_soft += dice_soft_iter
    Dice_each += np.array(dice_each_iter)

    sample_index += 1
    print(sample_index)

metrics_dict = {'GED': GED_global / len(all_slices_names),
                'Dice_max': Dice_max / len(all_slices_names),
                'Dice_max_reverse': Dice_max_reverse / len(all_slices_names),
                'Dice_max_mean': (Dice_max_reverse + Dice_max) / (2 * len(all_slices_names)),
                'Dice_match': Dice_match / len(all_slices_names),
                'Dice_soft': Dice_soft / len(all_slices_names),
                'Dice_each': Dice_each / len(all_slices_names),
                'Dice_each_mean': np.mean(Dice_each) / len(all_slices_names)}

for key in metrics_dict.keys():
    print(key, ': ', metrics_dict[key])