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
from monai.utils import set_determinism
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
    SpatialPadd,
    EnsureChannelFirstd,
)

set_determinism(seed=0)

data_dir = r"/home/kevin/anaconda3/envs/pytorch/Task07_Pancreas"
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
VAL_AMP = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    features=(16, 32, 64, 128, 256),
    kernel_size=(3, 3),     # Two Convolution for each layer. e.g. kernel_size=(3, 5) 1. used for 3*3, 2. used for 5*5
    dilation=(1, 1),        # # Two Convolution for each layer. e.g. dilation=(1, 2) 1. used for 1 and 2. used for 2
    upsample='deconv',
).to(device)

dice_metric = DiceMetric(include_background=True, reduction='mean')

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            input,
            roi_size=(200, 200, 40),
            sw_batch_size=2,
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
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[181:281], train_labels[181:281])
]

val_org_transforms = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image'], pixdim=[1.2, 1.1, 1.6], mode=('bilinear')),
        Orientationd(keys=['image'], axcodes='RAS'),
        CropForegroundd(keys=['image'], source_key='image'),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-96,
            a_max=215,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        # SpatialPadd(keys=['image', 'label'], spatial_size=[224, 224, 40]),
        # NormalizeIntensityd(keys=['image'], nonzero=False),
    ]
)
val_org_ds = Dataset(data=test_dicts, transform=val_org_transforms)

'''
for i in range(4):
    print(f"image shape: {val_org_ds[i]['image'].shape}")
    print(f"image shape: {val_org_ds[i]['label'].shape}")
'''

val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=1)
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
            device=torch.device('cpu'),
        ),
        Activationsd(keys='pred', softmax=True),
        AsDiscreted(keys='pred', argmax=True, to_onehot=None),
        AsDiscreted(keys='label', to_onehot=None),
        # SaveImaged(keys='pred', output_dir='./output', output_postfix='pred_seg', resample=False),
    ]
)

# val_post_trans = Compose(
#     [
#         Invertd(
#             keys='pred',
#             transform=val_org_transforms,
#             orig_keys='image',
#             meta_keys='pred_meta_dict',
#             orig_meta_keys='image_meta_dict',
#             meta_key_postfix='meta_dict',
#             nearest_interp=False,
#             to_tensor=True,
#             device=torch.device('cpu'),
#         ),
#         Activationsd(keys='pred', softmax=True),
#         AsDiscreted(keys='pred', argmax=True, to_onehot=3),
#         AsDiscreted(keys='label', to_onehot=3),
#         SaveImaged(keys='pred', output_dir='./output', output_postfix='pred_seg', resample=False),
#     ]
# )

model.load_state_dict(torch.load(os.path.join(data_dir, 'best_metric_model.pth')))
model.eval()

with torch.no_grad():
    for i, val_data in enumerate(val_org_loader):
        val_inputs = val_data['image'].to(device)
        val_data['pred'] = inference(val_inputs)
        torch.cuda.empty_cache()
        print(val_data['image'].shape)
        # print(val_data['pred'].shape)
        val_data['image'] = val_data['image'].to(torch.device('cpu'))
        val_data['pred'] = val_data['pred'].to(torch.device('cpu'))
        val_data = [val_post_trans(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(['pred', 'label'])(val_data)
        # print(val_outputs[0].shape)
        #
        # plt.figure("check image", (24, 8))
        # plt.subplot(1, 3, 1)
        # plt.title(f"image {i}")
        # plt.imshow(val_inputs[0, 0, :, :, 65].detach().cpu(), cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.title(f"label {i}")
        # plt.imshow(val_labels[0][0, :, :, 65])
        # plt.subplot(1, 3, 3)
        # plt.title(f"pred {i}")
        # plt.imshow(val_outputs[0][0, :, :, 65])
        # plt.show()
        dice_metric(y_pred=val_outputs, y=val_labels)
    metric_org = dice_metric.aggregate().item()
    dice_metric.reset()

print("metric on original image spacing: ", metric_org)