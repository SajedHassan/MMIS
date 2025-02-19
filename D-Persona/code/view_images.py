import h5py
import numpy as np

h5file = h5py.File('/home/sajed/thesis/MMIS/dataset/ORG_1_ONLY/training_2d/Sample_0_slice_17.h5', "r")

t1 = np.array(h5file['t1'])
t2 = np.array(h5file['t2'])
t1c = np.array(h5file['t1c'])
label1 = np.array(h5file['label_a1'])
label2 = np.array(h5file['label_a2'])
label3 = np.array(h5file['label_a3'])
label4 = np.array(h5file['label_a4'])

h5file.close()

