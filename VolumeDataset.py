import math
import os

import nibabel as nib
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from ScanSequence import Sequence

SLICE_COUNT = 154


def scale_data(input_data):
    min_val = np.min(input_data)
    max_val = np.max(input_data)
    if max_val == 0:
        return input_data
    scaled_data = (input_data - min_val) / (max_val - min_val)
    return scaled_data


def binarize_data(input_data):
    return input_data > 0


class VolumeDataset(Dataset):
    def __init__(self, path, transform=False, target_transform=None):
        self.path = path
        self.transform = transform
        if transform:
            self.transform_function = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

    def __len__(self):
        files = os.listdir("archive/BraTS2020_{0}Data/MICCAI_BraTS2020_{0}Data".format(self.path))
        volumes = set()
        for file in files:
            split = file.split("_")
            if split[0] == "BraTS20":
                volumes.add(split[2])
        return len(volumes) * SLICE_COUNT

    def __getitem__(self, idx):
        volume_int = math.ceil(idx / 155)
        if volume_int < 1:
            volume_int = 1
        target_volume = str(volume_int).zfill(3)
        target_slice = idx % 155
        target_sequence = [Sequence.T1, Sequence.T2, Sequence.FLAIR, Sequence.SEG]
        images = np.zeros((4, 240, 240))
        for idx, seq in enumerate(target_sequence):
            img = nib.load("archive/BraTS2020_{2}Data/MICCAI_BraTS2020_{2}Data/BraTS20_{2}_{0}/"
                           "BraTS20_{2}_{0}_{1}.nii"
                           .format(target_volume, seq.value, self.path))
            image = img.get_fdata()[:, :, target_slice]
            images[idx] = image
        train_images = images[0:3, :, :]
        seg_image = images[3, :, :]

        if self.transform:
            return np.float32(scale_data(train_images)), np.float32(binarize_data(np.array([seg_image])))
        return train_images, seg_image

    def get_item(self, idx):
        return self.__getitem__(idx)

    def get_len(self):
        return self.__len__()
