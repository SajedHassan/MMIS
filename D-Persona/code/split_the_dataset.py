import torch
import numpy as np
from dataloader.data_loader import DatasetSpliter
import os
import argparse

from configs.config import *
from lib.metrics_set import *
from dataloader.utils import data_preprocess
from tqdm import tqdm
import pickle
import h5py


path = '/home/sajed/thesis/MMIS/dataset/LIDC/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/home/sajed/thesis/MMIS/D-Persona/code/configs/params_lidc_split.yaml', help="config path (*.yaml)")
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    opt = Config(config_path=args.config)

    # dataset
    data_spliter = DatasetSpliter(opt=opt, input_size=opt.INPUT_SIZE)

    # Open the file in read-binary mode
    with open(opt.DATA_PATH, 'rb') as f:
        pkl_data = pickle.load(f)

    for fold_idx in range(opt.KFOLD):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        print('#********{} of {} FOLD *******#'.format(fold_idx+1, opt.KFOLD))
        train_loader, test_loader = data_spliter.get_datasets(fold_idx=fold_idx)

        train_path = path + 'FOLD_' + str(fold_idx + 1) + '/' + 'train/'
        val_path = path + 'FOLD_' + str(fold_idx + 1) + '/' + 'val/'

        for step, (patch, masks, sid) in enumerate(tqdm(train_loader)):
            patch, masks = data_preprocess(patch, masks, training=False)

            os.makedirs(train_path, exist_ok=True)
            slice = h5py.File(train_path + sid[0] + '.h5', "w")

            slice.create_dataset("image", data=patch[0][0])

            slice.create_dataset("label_a1", data=masks[0][0])
            slice.create_dataset("label_a2", data=masks[0][1])
            slice.create_dataset("label_a3", data=masks[0][2])
            slice.create_dataset("label_a4", data=masks[0][3])

            slice.close()

        for step, (patch, masks, sid) in enumerate(tqdm(test_loader)):
            patch, masks = data_preprocess(patch, masks, training=False)

            os.makedirs(val_path, exist_ok=True)
            slice = h5py.File(val_path + sid[0] + '.h5', "w")

            slice.create_dataset("image", data=patch[0][0])

            slice.create_dataset("label_a1", data=masks[0][0])
            slice.create_dataset("label_a2", data=masks[0][1])
            slice.create_dataset("label_a3", data=masks[0][2])
            slice.create_dataset("label_a4", data=masks[0][3])

            slice.close()
