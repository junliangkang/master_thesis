import os
import glob
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.colors import ListedColormap
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    SaveImaged,
    EnsureTyped,
    EnsureChannelFirstd,
)



dir = '/home/kevin/anaconda3/envs/pytorch/Task03_Liver'
seg_dir = '/home/kevin/PycharmProjects/pythonProject'
main_img = nib.load(os.path.join(dir, "imagesTr", "liver_6.nii.gz"))
label_img = nib.load(os.path.join(dir, "labelsTr", "liver_6.nii.gz"))
seg_img = nib.load(os.path.join(seg_dir, "output", "liver_6", "liver_6_pred_seg.nii.gz"))
print(f'original image shape is: {main_img.shape}')
print(f'label shape is: {label_img.shape}')
print(f'output pred shape is: {seg_img.shape}')

main_img_data = main_img.get_fdata()
label_img_data = label_img.get_fdata()
seg_img_data = seg_img.get_fdata()

# cmap = ListedColormap(['purple', 'cyan', 'yellow'])

plt.figure("result", (24, 8))
plt.subplot(1, 3, 1)
plt.title("original image")
plt.imshow(main_img_data[:, :, 472], cmap="gray")
plt.subplot(1, 3, 2)
plt.title('label')
plt.imshow(label_img_data[:, :, 472])
plt.subplot(1, 3, 3)
plt.title('output')
plt.imshow(seg_img_data[:, :, 472])
plt.show()




# data_dir = r"/home/kevin/anaconda3/envs/pytorch/Task03_Liver"
# seg_dir = r'/home/kevin/PycharmProjects/pythonProject'
# val_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
# val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
# val_dicts = [
#     {"image": image_name, "label": label_name} for image_name, label_name in zip(val_images[-8:], val_labels[-8:])
# ]
# val_transform = Compose(
#     [
#         LoadImaged(keys=['image', 'label']),
#         EnsureChannelFirstd(keys=['image', 'label']),
#         EnsureTyped(keys=['image', 'label']),
#     ]
# )
# print(val_dicts[0:8])
# print(len(val_dicts))
# val_ds = Dataset(data=val_dicts, transform=val_transform)
# data_example = val_ds[2]
# print(f"image shape: {data_example['image'].shape}")
# print(f"label shape: {data_example['label'].shape}")
# plt.figure("image", (18, 6))
# plt.subplot(1, 2, 1)
# plt.title('image')
# plt.imshow(data_example['image'][0, :, :, 550].detach().cpu(), cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title('label')
# plt.imshow(data_example['label'][0, :, :, 550].detach().cpu())
# plt.show()