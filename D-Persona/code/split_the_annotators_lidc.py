import os
import shutil

# Path to your folder containing .h5 files
path = '/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_4/'
type = 'train'

# List all .h5 files
files = [f for f in os.listdir(path + type) if f.endswith(".h5")]

# Extract patient IDs and count slices per patient
patient_slices = {}
for f in files:
    patient_id = f.split("_slice")[0]
    patient_slices.setdefault(patient_id, []).append(f)

# Sort patients by number of slices (descending)
sorted_patients = sorted(patient_slices.items(), key=lambda x: len(x[1]), reverse=True)

# Initialize 4 groups
groups = [[] for _ in range(4)]
group_sizes = [0] * 4  # Track total slices per group

# Distribute patients into groups (greedy balancing)
for patient_id, slices in sorted_patients:
    min_index = group_sizes.index(min(group_sizes))  # Find the group with the least slices
    groups[min_index].extend(slices)  # Assign slices to that group
    group_sizes[min_index] += len(slices)  # Update count

# Optional: Move files into separate folders for each group
for i, group in enumerate(groups):
    group_folder = os.path.join(path, f"splitted/{type}_with_all_ann/a{i}")
    os.makedirs(group_folder, exist_ok=True)
    for file_name in group:
        shutil.copy(os.path.join(path, type, file_name), os.path.join(group_folder, file_name))

print("Files have been split into 4 balanced groups.")
