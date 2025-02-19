import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from skimage import exposure
import nibabel as nib

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        modality="t1c",
        transform=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.modality = modality

        if self.split == "train":
            self.sample_list += [f for f in os.listdir(self._base_dir + "/training/Annotator_all") if not f.startswith('.')]
        elif self.split == "val":
            self.sample_list += [f for f in os.listdir(self._base_dir + "/validation/Annotator_all") if not f.startswith('.')]
        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/testing")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            # Load the NIfTI file
            t1 = nib.load(self._base_dir +"/training/Annotator_all/{}/{}".format(case, 'T1.nii.gz')).get_fdata()
            t2 = nib.load(self._base_dir +"/training/Annotator_all/{}/{}".format(case, 'T2.nii.gz')).get_fdata()
            ct1 = nib.load(self._base_dir +"/training/Annotator_all/{}/{}".format(case, 'CT1.nii.gz')).get_fdata()
            flair = nib.load(self._base_dir +"/training/Annotator_all/{}/{}".format(case, 'FLAIR.nii.gz')).get_fdata()
            label = nib.load(self._base_dir +"/training/Annotator_all/{}/{}".format(case, 'CT1_seg.nii.gz')).get_fdata()
        elif self.split == "val":
            t1 = nib.load(self._base_dir +"/validation/Annotator_all/{}/{}".format(case, 'T1.nii.gz')).get_fdata()
            t2 = nib.load(self._base_dir +"/validation/Annotator_all/{}/{}".format(case, 'T2.nii.gz')).get_fdata()
            ct1 = nib.load(self._base_dir +"/validation/Annotator_all/{}/{}".format(case, 'CT1.nii.gz')).get_fdata()
            flair = nib.load(self._base_dir +"/validation/Annotator_all/{}/{}".format(case, 'FLAIR.nii.gz')).get_fdata()
            label = nib.load(self._base_dir +"/validation/Annotator_all/{}/{}".format(case, 'CT1_seg.nii.gz')).get_fdata()
        elif self.split == "test":
            t1 = nib.load(self._base_dir +"/testing/Annotator_all/{}/{}".format(case, 'T1.nii.gz')).get_fdata()
            t2 = nib.load(self._base_dir +"/testing/Annotator_all/{}/{}".format(case, 'T2.nii.gz')).get_fdata()
            ct1 = nib.load(self._base_dir +"/testing/Annotator_all/{}/{}".format(case, 'CT1.nii.gz')).get_fdata()
            flair = nib.load(self._base_dir +"/testing/Annotator_all/{}/{}".format(case, 'FLAIR.nii.gz')).get_fdata()
            label = nib.load(self._base_dir +"/testing/Annotator_all/{}/{}".format(case, 'CT1_seg.nii.gz')).get_fdata()

        # image = h5f[self.modality][:]
        # image_modality_list = ["t1", "t1c", "t2"]
        # image = np.array([h5f[modality][:] for modality in image_modality_list])

        if self.split == "train":
            t1_slices_axis_0, t1_slices_axis_1, t1_slices_axis_2 = get_2d_slices_all_axes(t1)
            ct1_slices_axis_0, ct1_slices_axis_1, ct1_slices_axis_2 = get_2d_slices_all_axes(ct1)
            t2_slices_axis_0, t2_slices_axis_1, t2_slices_axis_2 = get_2d_slices_all_axes(t2)
            flair_slices_axis_0, flair_slices_axis_1, flair_slices_axis_2 = get_2d_slices_all_axes(flair)
            label_slices_axis_0, label_slices_axis_1, label_slices_axis_2 = get_2d_slices_all_axes(label)

            images_axis_0 = combine_to_channels_first(t1_slices_axis_0, ct1_slices_axis_0, t2_slices_axis_0,
                                                      ct1_slices_axis_0.shape)
            images_axis_1 = combine_to_channels_first(t1_slices_axis_1, ct1_slices_axis_1, t2_slices_axis_1,
                                                      ct1_slices_axis_1.shape)
            images_axis_2 = combine_to_channels_first(t1_slices_axis_2, ct1_slices_axis_2, t2_slices_axis_2,
                                                      ct1_slices_axis_2.shape)

            label_axis_0 = np.array([[l, l, l, l] for l in label_slices_axis_0])
            label_axis_1 = np.array([[l, l, l, l] for l in label_slices_axis_1])
            label_axis_2 = np.array([[l, l, l, l] for l in label_slices_axis_2])

            try:
                samples = []
                for i in range(0, images_axis_0.shape[0]):
                    if (images_axis_0[i] == 0).all():
                        continue
                    sample = {"image": images_axis_0[i], "label": label_axis_0[i]}
                    sample = self.transform(sample)
                    sample["idx"] = case + '_slice_' + str(i)
                    samples.append(sample)

                for i in range(0, images_axis_1.shape[0]):
                    if (images_axis_1[i] == 0).all():
                        continue
                    sample = {"image": images_axis_1[i], "label": label_axis_1[i]}
                    sample = self.transform(sample)
                    sample["idx"] = case + '_slice_' + str(i)
                    samples.append(sample)

                for i in range(0, images_axis_2.shape[0]):
                    if (images_axis_2[i] == 0).all():
                        continue
                    sample = {"image": images_axis_2[i], "label": label_axis_2[i]}
                    sample = self.transform(sample)
                    sample["idx"] = case + '_slice_' + str(i)
                    samples.append(sample)

                if np.array([samples[i]['image'].shape == torch.Size([3, 128, 128]) for i in range(0, len(samples))]).all():
                    return samples
                else:
                    print('error3')
                    raise 'error3'
            except Exception as e:
                print('error2')
                raise e
        else:
            t1 = t1 if t1.shape == ct1.shape else ct1
            t2 = t2 if t2.shape == ct1.shape else ct1
            image = np.array([t1, ct1, t2])
            label = np.array([label, label, label, label])
            try:
                sample = {"image": image, "label": label}
                sample = self.transform(sample)

                sample["idx"] = case
                return sample
            except Exception as e:
                print('error2')
                raise e

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)

    if len(image.shape) == 2:
        image = np.rot90(image, k)
        image = np.flip(image, axis=axis).copy()
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], k)
            image[i] = np.flip(image[i], axis=axis).copy()
    if len(label.shape) == 2:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = np.rot90(label[i], k)
            label[i] = np.flip(label[i], axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if len(image.shape) == 2:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = ndimage.rotate(image[i], angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = ndimage.rotate(label[i], angle, order=0, reshape=False)
    return image, label

def random_noise(image, label, mu=0, sigma=0.1):
    if len(image.shape) == 2:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]), -2 * sigma, 2 * sigma)
    elif len(image.shape) == 3:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * sigma, 2 * sigma)
    else:
        pass
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label

def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label

def get_2d_slices_all_axes(image_3d):
    # Slices along axis 0 (depth)
    slices_axis_0 = np.array([slice_2d for slice_2d in image_3d])
    
    # Slices along axis 1 (height) -> Transpose axes (1, 0, 2)
    slices_axis_1 = np.array([slice_2d for slice_2d in np.transpose(image_3d, (1, 0, 2))])
    
    # Slices along axis 2 (width) -> Transpose axes (2, 0, 1)
    slices_axis_2 = np.array([slice_2d for slice_2d in np.transpose(image_3d, (2, 0, 1))])
    
    # Return the slices for all axes
    return slices_axis_0, slices_axis_1, slices_axis_2

def combine_to_channels_first(array_1, array_2, array_3, expected_shape):
    try:
        array_1 = array_1 if array_1.shape == expected_shape else np.copy(array_2)
        array_2 = array_2 # this always matches the shape of the label
        array_3 = array_3 if array_3.shape == expected_shape else np.copy(array_2)

        # Pad the arrays to the maximum shape to make sure all have the same shape
        # max_shape = np.max([array_1.shape, array_2.shape, array_3.shape], axis=0)
        # make sure the number of slices for each scan type is the same so we can combine them on the channel axis
        # [5, h, w] & [7, h, w] will both end up to [5, h, w]
        # min_size = min(arr.shape[0] for arr in [array_1, array_2, array_3])
        # padded_arrays = [
        #     np.pad(arr, 
        #         [(0, (expected_shape[dim] if expected_shape[dim] > arr.shape[dim] else arr.shape[dim]) - arr.shape[dim]) for dim in range(len(expected_shape))], 
        #         mode='constant') 
        #     for arr in [array_1, array_2, array_3]
        # ]
        # adjusted_arrays = [arr[:min_size] for arr in padded_arrays]
        # Stack the arrays along the first axis (axis=0 for channels)
        combined_array = np.stack((array_1, array_2, array_3), axis=1)

        # The result will be of shape (3, N, H, W), where 3 is the channel dimension
        return combined_array
    except ValueError as e:
        print('error')
        raise e

class RandomGenerator_Multi_Rater(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        try: 
            image, label = sample["image"], sample["label"]

            _, x, y = image.shape

            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            if len(label.shape) == 2:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            elif len(label.shape) == 3:
                label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            if random.random() > 0.5:
                image, label = random_rotate(image, label)
            if random.random() > 0.5:
                image, label = random_noise(image, label)
            # if random.random() > 0.5:
            #     image, label = random_rescale_intensity(image, label)
            # if random.random() > 0.5:
            #     image, label = random_equalize_hist(image, label)
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.uint8))
            sample = {"image": image, "label": label}
            return sample
        except Exception as e:
            print(e)
            raise e

class ZoomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        c, d, x, y = image.shape

        image = zoom(image, (1, 1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (1, 1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
