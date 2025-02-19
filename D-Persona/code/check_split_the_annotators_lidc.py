import os
import shutil

splits = []
for i in range(4):
    type = 'train'
    path = os.path.join('/home/sajed/thesis/MMIS/dataset/LIDC/FOLD_4/', f"splitted/{type}_with_all_ann/a{i}")
    files = [f for f in os.listdir(path) if f.endswith(".h5")]
    patient_slices = {}
    for f in files:
        patient_id = f.split("_slice")[0]
        patient_slices.setdefault(patient_id, []).append(f)
    splits.append(patient_slices)

print(
    len((splits[0].keys() & splits[1].keys())) == 0 and
    len((splits[0].keys() & splits[2].keys())) == 0 and
    len((splits[0].keys() & splits[3].keys())) == 0 and

    len((splits[1].keys() & splits[2].keys())) == 0 and
    len((splits[1].keys() & splits[3].keys())) == 0 and

    len((splits[2].keys() & splits[3].keys())) == 0 
)