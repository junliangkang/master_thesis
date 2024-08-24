import nibabel as nib
import glob
import os


dir = '/home/kevin/anaconda3/envs/pytorch/Task07_Pancreas'
main_img = nib.load(os.path.join(dir, "imagesTr", "liver_106.nii.gz"))
main_img_shape = main_img.shape
print(f"Main image shape: {main_img_shape}")


seg_img = nib.load('/home/kevin/PycharmProjects/pythonProject/output/liver_106/liver_106_pred_seg.nii.gz')
seg_img_shape = seg_img.shape
print(f"Segmentation image shape: {seg_img_shape}")
