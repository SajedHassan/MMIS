import os
import h5py
import numpy as np

original_dataset_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK1/training_2d/'
generated_dataset_dir = '/home/sajed/thesis/MMIS/dataset/GENERATED_NPC_FROM_NNUNET_WITH_LEARNABLE_EMB_E32_SM_ES_DS_F5_P250_(2)/training_2d/'
split_dirs = [
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a0/',
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a1/',
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a2/',
    '/home/sajed/thesis/MMIS/dataset/splitted/training_2d/a3/'
]
number_of_ann = len(split_dirs)

for annotator_index, split_dir in enumerate(split_dirs):
    samples_names = [f for f in os.listdir(split_dir) if not f.startswith('.')]

    print(len(samples_names))

    valid_samples = []

    for sample_name in samples_names:
        original_sample = h5py.File(original_dataset_dir + sample_name, 'r')
        generated_sample = h5py.File(generated_dataset_dir + sample_name, 'r')

        original_t1 = np.array(original_sample['t1'])
        original_t1c = np.array(original_sample['t1c'])
        original_t2 = np.array(original_sample['t2'])
        original_label_a1 = np.array(original_sample['label_a1'])
        original_label_a2 = np.array(original_sample['label_a2'])
        original_label_a3 = np.array(original_sample['label_a3'])
        original_label_a4 = np.array(original_sample['label_a4'])
        original_labels = [original_label_a1, original_label_a2, original_label_a3, original_label_a4]

        generated_t1 = np.array(generated_sample['t1'])
        generated_t1c = np.array(generated_sample['t1c'])
        generated_t2 = np.array(generated_sample['t2'])
        generated_label_a1 = np.array(generated_sample['label_a1'])
        generated_label_a2 = np.array(generated_sample['label_a2'])
        generated_label_a3 = np.array(generated_sample['label_a3'])
        generated_label_a4 = np.array(generated_sample['label_a4'])
        generated_labels = [generated_label_a1, generated_label_a2, generated_label_a3, generated_label_a4]

        valid = (
            (original_t1 == generated_t1).all() and
            (original_t1c == generated_t1c).all() and
            (original_t2 == generated_t2).all() and
            (original_labels[annotator_index] == generated_labels[annotator_index]).all() and
            (original_labels[(annotator_index + 1) % number_of_ann] == generated_labels[(annotator_index + 1) % number_of_ann]).all() and
            # (original_labels[annotator_index] == generated_labels[(annotator_index + 2) % number_of_ann]).all() and
            # (original_labels[annotator_index] == generated_labels[(annotator_index + 3) % number_of_ann]).all() and
            original_t1.shape == original_t1.shape ==
            original_t1c.shape == generated_t1c.shape ==
            original_t2.shape == generated_t2.shape ==
            original_label_a1.shape == generated_label_a1.shape ==
            original_label_a2.shape == generated_label_a2.shape ==
            original_label_a3.shape == generated_label_a3.shape ==
            original_label_a4.shape == generated_label_a4.shape
        )
        for index in [0,1,2,3]:
            # if index != annotator_index and index != (annotator_index + 1) % number_of_ann and index != (annotator_index + 2) % number_of_ann and index != (annotator_index + 3) % number_of_ann:
            if index != annotator_index and index != (annotator_index + 1) % number_of_ann:
                valid = valid and (
                    ((original_labels[annotator_index] != generated_labels[index]).any() or (generated_labels[index] == 0).all()) and
                    ((original_labels[index] != generated_labels[index]).any() or (generated_labels[index] == 0).all())
                )

        valid_samples.append(valid)

        original_sample.close()
        generated_sample.close()

    valid_samples = np.array(valid_samples)
    print(len(valid_samples))
    print((valid_samples == True).all())