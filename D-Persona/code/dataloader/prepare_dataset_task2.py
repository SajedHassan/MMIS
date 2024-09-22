# import os

# # Specify the directory and the prefix to remove
# base_dir = '/Users/sajedalmorsy/Academic/Masters/thesis/D-Persona/dataset/MMIS2024TASK2_train/validation/Annotator_all'
# samples_directories = [f for f in os.listdir(base_dir) if not f.startswith('.')]
# sub_dataset_source = 'RHUH'

# for sample_directory in samples_directories:
#     if sample_directory.startswith(sub_dataset_source):
#       for filename in os.listdir(base_dir + '/' + sample_directory):

#         old_file = base_dir + '/' + os.path.join(sample_directory, filename)
#         new_file = base_dir + '/'

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

#         # Rename the file
#         os.rename(old_file, new_file)
#         # print(old_file, new_file)

# print("Corrected files names")









# import os

# # Specify the directory and the prefix to remove
# base_dir = '/Users/sajedalmorsy/Academic/Masters/thesis/D-Persona/dataset/MMIS2024TASK2_train/validation/Annotator_all'
# samples_directories = [f for f in os.listdir(base_dir) if not f.startswith('.')]
# sub_dataset_source = 'UPENN'

# for sample_directory in samples_directories:
#     if sample_directory.startswith(sub_dataset_source):
#       for filename in os.listdir(base_dir + '/' + sample_directory):

#         old_file = base_dir + '/' + os.path.join(sample_directory, filename)
#         new_file = base_dir + '/'

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

#         # Rename the file
#         os.rename(old_file, new_file)
#         # print(old_file, new_file)

# print("Corrected files names")







a = [i for i in range(0, 33, 5)]
print(a)