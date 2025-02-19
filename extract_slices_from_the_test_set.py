import os
import h5py
import numpy as np

set_base_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/validation/'
target_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/validation_2d/'

samples_names = [f for f in os.listdir(set_base_dir) if not f.startswith('.')]

for sample_name in samples_names:
    sample = h5py.File(set_base_dir + sample_name, 'r')
    t1 = np.array(sample['t1'])
    t1c = np.array(sample['t1c'])
    t2 = np.array(sample['t2'])

    label_a1 = np.array(sample['label_a1'])
    label_a2 = np.array(sample['label_a2'])
    label_a3 = np.array(sample['label_a3'])
    label_a4 = np.array(sample['label_a4'])

    for idx in range(0, t1.shape[0]):
        t1_2d = t1[idx]
        t1c_2d = t1c[idx]
        t2_2d = t2[idx]

        label_a1_2d = label_a1[idx]
        label_a2_2d = label_a2[idx]
        label_a3_2d = label_a3[idx]
        label_a4_2d = label_a4[idx]

        slice_name = sample_name.replace('.h5', '') + '_slice_' + str(idx) + '.h5'
        slice = h5py.File(os.path.join(target_dir, slice_name), "w")

        slice.create_dataset("t1", data=t1_2d)
        slice.create_dataset("t1c", data=t1c_2d)
        slice.create_dataset("t2", data=t2_2d)

        slice.create_dataset("label_a1", data=label_a1_2d)
        slice.create_dataset("label_a2", data=label_a2_2d)
        slice.create_dataset("label_a3", data=label_a3_2d)
        slice.create_dataset("label_a4", data=label_a4_2d)

        slice.close()

    sample.close()

