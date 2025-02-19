# import os
#
# # Specify the directory and the prefix to remove
# base_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK2_train/training/Annotator_all'
# samples_directories = [f for f in os.listdir(base_dir) if not f.startswith('.')]
# sub_dataset_source = 'RHUH'
#
# for sample_directory in samples_directories:
#     if sample_directory.startswith(sub_dataset_source):
#       for filename in os.listdir(base_dir + '/' + sample_directory):
#
#         old_file = base_dir + '/' + os.path.join(sample_directory, filename)
#         new_file = base_dir + '/'
#
#         if 't1.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'T1.nii.gz')
#         elif 't2.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'T2.nii.gz')
#         elif 't1ce.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'CT1.nii.gz')
#         elif 't1ce_seg' in filename:
#           new_file += os.path.join(sample_directory, 'CT1_seg.nii.gz')
#         elif 'flair.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'FLAIR.nii.gz')
#         else:
#           print('FILENAME: ', filename)
#           continue
#
#         # Rename the file
#         os.rename(old_file, new_file)
#         # print(old_file, new_file)
#
# print("Corrected files names")







#
#
# import os
#
# # Specify the directory and the prefix to remove
# base_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK2_train/training/Annotator_all'
# samples_directories = [f for f in os.listdir(base_dir) if not f.startswith('.')]
# sub_dataset_source = 'UPENN-GBM-00134_21'
#
# for sample_directory in samples_directories:
#     if sample_directory.startswith(sub_dataset_source):
#       for filename in os.listdir(base_dir + '/' + sample_directory):
#
#         old_file = base_dir + '/' + os.path.join(sample_directory, filename)
#         new_file = base_dir + '/'
#
#         if 'T1.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'T1.nii.gz')
#         elif 'T2.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'T2.nii.gz')
#         elif 'T1GD.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'CT1.nii.gz')
#         elif 'T1GD_seg.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'CT1_seg.nii.gz')
#         elif 'FLAIR.nii.gz' in filename:
#           new_file += os.path.join(sample_directory, 'FLAIR.nii.gz')
#         else:
#           print('FILENAME: ', filename)
#           continue
#
#         # Rename the file
#         os.rename(old_file, new_file)
#         # print(old_file, new_file)
#
# print("Corrected files names")




# import os

# # Specify the directory and the prefix to remove
# base_dir = '/home/sajed/thesis/MMIS/dataset/MMIS2024TASK2_train/training/Annotator_all'
# samples_directories = [f for f in os.listdir(base_dir) if not f.startswith('.')]
# sub_dataset_source = ''

# for sample_directory in samples_directories:
#   for filename in os.listdir(base_dir + '/' + sample_directory):

#     if 'CT1_seg.gz' in filename:
#       print('FILENAME: ', sample_directory)




# import os
# import numpy as np
# import h5py
# import re

# base_dir = '/home/sajed/thesis/MMIS/dataset/splitted/validation_v2'
# slices = [f for f in os.listdir(base_dir) if not f.startswith('.')]

# print(len(slices))
# index = 0
# for slice in slices:
#   # Open the file, create a new dataset, and delete an existing one
#   with h5py.File(os.path.join(base_dir, slice), 'a') as f:
#       # print(f['label_a4'].shape)
#       group = int(index / 5) + 1
#       label_key = 'label_a' + str(group)
#       # print(label_key)
#       # f.create_dataset('label', data=f[label_key])
#       # del f['label_a1']
#       # del f['label_a2']
#       # del f['label_a3']
#       # del f['label_a4']
#       index += 1

# print(len(slices))
# f = h5py.File(os.path.join(base_dir, slices[0]), "r")
# print(f['label'].shape)




# import os
# import numpy as np
# import h5py
# import re
# from PIL import Image

# src_dir = '/home/sajed/thesis/stargan-v2/expr/results/90000/a0_to_a0/'
# target_dir = '/home/sajed/thesis/MMIS/dataset/splitted/training_2d_v2_star/'
# image_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]

# # print(len(slices))
# # index = 0
# for image_file in image_files:
#   img1_path = '/home/sajed/thesis/stargan-v2/expr/results/90000/a0_to_a0/' + image_file
#   img2_path = '/home/sajed/thesis/stargan-v2/expr/results/90000/a0_to_a1/' + image_file
#   img3_path = '/home/sajed/thesis/stargan-v2/expr/results/90000/a0_to_a2/' + image_file
#   img4_path = '/home/sajed/thesis/stargan-v2/expr/results/90000/a0_to_a3/' + image_file

#   img1 = Image.open(img1_path)
#   img2 = Image.open(img1_path)
#   img3 = Image.open(img1_path)
#   img3 = Image.open(img1_path)
  
  # with h5py.File(os.path.join(base_dir, slice), 'a') as f:
  #     # print(f['label_a4'].shape)
  #     group = int(index / 5) + 1
  #     label_key = 'label_a' + str(group)
  #     # print(label_key)
  #     # f.create_dataset('label', data=f[label_key])
  #     # del f['label_a1']
  #     # del f['label_a2']
  #     # del f['label_a3']
  #     # del f['label_a4']
  #     index += 1

# print(len(slices))
# f = h5py.File(os.path.join(base_dir, slices[0]), "r")
# print(f['label'].shape)



# check that the validation data is correct
import os
import numpy as np
import h5py
import re

base_dir = '/home/sajed/thesis/MMIS/dataset/GENERATED_NPC/validation/'
samples_files = [f for f in os.listdir(base_dir) if not f.startswith('.')]

# print(len(slices))
# index = 0
for image_file in samples_files:
  sample_path = base_dir + image_file
  f = h5py.File(sample_path, 'r')
  print(f.keys())
  f.close()
