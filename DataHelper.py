import os

import torch
import matplotlib.pyplot as plt
import nibabel as nib

from ScanSequence import Sequence
from VolumeDataset import VolumeDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_stack(target_volume, sequence: Sequence):
    target_volume = str(target_volume)
    target_volume_pad = target_volume.zfill(3)
    img = nib.load(
        "archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_{1}.nii"
        .format(target_volume_pad, sequence.value))
    return img


def load_volume_set(sequence: Sequence):
    files = os.listdir("BraTS h5py/BraTS2020_training_data/content/data")
    volumes = set()
    for file in files:
        split = file.split("_")[1]
        volumes.add(split)
    return volumes


def preview_volume_slice(volume_num: int, slice_num: int):
    image_sequences = [Sequence.T1CE, Sequence.T2, Sequence.FLAIR, Sequence.SEG]

    fig = plt.figure(figsize=(8, 8))

    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        ax.title.set_text(image_sequences[i-1].value)
        img = load_stack(volume_num, image_sequences[i-1])
        plt.imshow(img.get_fdata()[:, :, slice_num])
    plt.show()


def preview_dataset(idx):
    # Data from: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?rvi=1
    vd = VolumeDataset("Training", transform=True)
    images, seg = vd.get_item(idx)

    print("Seg Max: {}".format(seg.max()))
    print("Image Max: {}".format(images.max()))

    fig = plt.figure(figsize=(8, 8))
    for i in range(1, 4):
        fig.add_subplot(2, 2, i)
        plt.imshow(images[i - 1])
    fig.add_subplot(2, 2, 4)
    plt.imshow(seg[0])
    plt.show()


if __name__ == "__main__":
    # Preview Sequence
    # preview_volume_slice(3, 78)
    preview_dataset(534)
