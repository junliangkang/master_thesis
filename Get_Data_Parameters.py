import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from monai.data import DataLoader, decollate_batch
from monai.config import print_config
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    Orientationd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
def get_voxel_spacing(nifti_file):
    image = nib.load(nifti_file)
    header = image.header
    spacing = header.get_zooms()  # get voxel spacing
    return spacing


if __name__ == "__main__":
    print("-" * 30)
    print_config()
    print("-" * 30)
    print("processing")
    data_dir = r"/home/kevin/anaconda3/envs/pytorch/Task07_Pancreas"
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[:20], train_labels[:20])
    ]
    train_data = data_dicts[:20]
    val_data = data_dicts[-10:]
    loader = LoadImaged(keys=["image", "label"])

    for i in range(20):
        check_data = loader(train_data[i])
        print(f"voxel spacing is: {get_voxel_spacing(train_data[i]['image'])}")
        print(f"image shape: {check_data['image'].shape}")
        print(f"image shape: {check_data['label'].shape}")



    loader = LoadImaged(keys=["image", "label"])
    check_data = loader(train_data[1])
    print(check_data)
    print(f"image shape: {check_data["image"].shape}")
    print(f"label shape: {check_data["label"].shape}")
    print(check_data.keys())
    print(check_data['image'][0])
    print(check_data['image'][1])
    print(f"Image data type: {type(check_data['image'])}")
    print(f"Label data type: {type(check_data['label'])}")
    print(f"Image data range: {np.min(check_data['image'])} to {np.max(check_data['image'])}")
    print(f"Label data range: {np.min(check_data['label'])} to {np.max(check_data['label'])}")
    # visualization
    image = check_data['image']
    label = check_data['label']
    plt.figure('visualize',(16, 8))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 65], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('label')
    plt.imshow(label[:, :, 65])
    plt.show()


