import torch
import os
import glob
import matplotlib.pyplot as plt
from monai.data import Dataset
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets.modunet import ModUNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    SaveImaged,
    EnsureChannelFirstd,
)


data_dir = r"/home/kevin/anaconda3/envs/pytorch/Task02_Heart"
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
train_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[:8], train_labels[:8])
]
val_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[-8:], train_labels[-8:])

]
print(train_dicts)
print(val_dicts)

VAL_AMP = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    features=(16, 32, 64),
    kernel_size=3,
    dilation=1,
    upsample='deconv',
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=True, reduction='mean')

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            input,
            roi_size=(192, 192, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

torch.backends.cudnn.benchmark = True
test_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[9:13], train_labels[9:13])
]

val_org_transforms = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image'], pixdim=[1.25, 1.25, 1.25], mode=('bilinear')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        NormalizeIntensityd(keys=['image'], nonzero=False),
        CropForegroundd(keys=['image'], source_key='image'),
    ]
)

val_org_ds = Dataset(data=test_dicts, transform=val_org_transforms)

'''
for i in range(4):
    print(f"image shape: {val_org_ds[i]['image'].shape}")
    print(f"image shape: {val_org_ds[i]['label'].shape}")
'''

val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4)
val_post_trans = Compose(
    [
        Invertd(
            keys='pred',
            transform=val_org_transforms,
            orig_keys='image',
            meta_keys='pred_meta_dict',
            orig_meta_keys='image_meta_dict',
            meta_key_postfix='meta_dict',
            nearest_interp=False,
            to_tensor=True,
            device='cpu',
        ),
        Activationsd(keys='pred', softmax=True),
        AsDiscreted(keys='pred', argmax=True, to_onehot=2),
        AsDiscreted(keys='label', to_onehot=2),
        SaveImaged(keys='pred', output_dir='./output', output_postfix='pred_seg', resample=False),
    ]
)

model.load_state_dict(torch.load(os.path.join(data_dir, 'best_metric_model.pth')))
model.eval()

with torch.no_grad():
    for i, val_data in enumerate(val_org_loader):
        val_inputs = val_data['image'].to(device)
        val_data['pred'] = inference(val_inputs)
        val_data = [val_post_trans(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(['pred','label'])(val_data)
        plt.figure("check image", (24, 8))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_inputs[0, 0, :, :, 65].detach().cpu(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_labels[0][0, :, :, 65])
        plt.subplot(1, 3, 3)
        plt.title(f"pred {i}")
        plt.imshow(val_outputs[0][0, :, :, 65])
        plt.show()
        dice_metric(y_pred=val_outputs, y=val_labels)
    metric_org = dice_metric.aggregate().item()
    dice_metric.reset()

print("metric on original image spacing: ", metric_org)