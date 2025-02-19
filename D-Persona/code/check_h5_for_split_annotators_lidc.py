import os
import shutil
import h5py
import numpy as np

checks = []
for i in range(4):
    type = 'train'
    ann_path = os.path.join('/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/', f"splitted/{type}/a{i}")
    ann_with_all_ann_path = os.path.join('/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_1/', f"splitted/{type}_with_all_ann/a{i}")
    for file_name in os.listdir(ann_path):
        if not file_name.endswith(".h5"):
            continue
        
        file_path = os.path.join(ann_path, file_name)
        file_with_all_ann_path = os.path.join(ann_with_all_ann_path, file_name)

        # Open the HDF5 file and modify it
        f = h5py.File(file_path, "r")
        f_with_all_ann = h5py.File(file_with_all_ann_path, "r")

        checks.append((np.array(f['label']) == np.array(f_with_all_ann['label_a' + str(i+1)])).all())

        f.close()
        f_with_all_ann.close()

print((np.array(checks) == True).all())