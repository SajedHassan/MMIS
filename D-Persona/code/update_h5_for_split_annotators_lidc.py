import os
import shutil
import h5py

for i in range(4):
    type = 'train'
    ann_path = os.path.join('/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_4/', f"splitted/{type}/a{i}")
    for file_name in os.listdir(ann_path):
        if not file_name.endswith(".h5"):
            continue
        
        file_path = os.path.join(ann_path, file_name)

        # Open the HDF5 file and modify it
        with h5py.File(file_path, "r+") as f:
            datasets = list(f.keys())  # Get all dataset names
            
            # Remove datasets that don't belong to this group
            dataset_to_keep = 'label_a' + str(i+1)
            for dataset in ['label_a1', 'label_a2', 'label_a3', 'label_a4']:
                if dataset != dataset_to_keep:
                    del f[dataset]

            # Rename the remaining dataset to "label"
            f.move(dataset_to_keep, "label")