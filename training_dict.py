import sys

import torch
import os
import glob
import time
import matplotlib.pyplot as plt
import logging
from monai.data import Dataset, CacheDataset
from monai.data import DataLoader, decollate_batch, pad_list_data_collate
from monai.config import print_config
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import basicunet
from monai.networks.nets.modunet import ModUNet
from monai.networks.layers import Norm
from monai.transforms import (
    Activationsd,
    Activations,
    AsDiscreted,
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandFlipd,
    Orientationd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    Spacingd,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    SaveImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    SpatialPadd,
)
from monai.utils import set_determinism

set_determinism(seed=0)

# logging.basicConfig(filename='Liver.log', level=logging.INFO)
# logging.info('This is an info message')
# logging.shutdown()

data_dir = r"/home/kevin/anaconda3/envs/pytorch/Task07_Pancreas"
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
train_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[:100], train_labels[:100])
]
val_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[100:181], train_labels[100:181])

]
print(train_dicts)
print(val_dicts)


# define transform for image and segmentation


train_transform = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=[1.2, 1.1, 1.6], mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        # NormalizeIntensityd(keys=['image'], nonzero=False),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-96,
            a_max=215,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        # SpatialPadd(keys=['image', 'label'], spatial_size=[128, 128, 64]),
        RandCropByPosNegLabeld(
            keys=['image','label'],
            label_key='label',
            spatial_size=[200, 200, 40],
            pos=2,
            neg=1,
            num_samples=3,
            image_key='image',
            image_threshold=0,
        ),
        RandZoomd(keys=['image', 'label'], min_zoom=0.9, max_zoom=1.2, mode=('trilinear', 'nearest'), align_corners=(True, None), prob=0.5),
        RandGaussianNoised(keys=['image'], std=0.01, prob=0.5),
        RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.5),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=['image'], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
        EnsureTyped(keys=['image', 'label']),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=[1.2, 1.1, 1.6], mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-96,
            a_max=215,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        # SpatialPadd(keys=['image', 'label'], spatial_size=[128, 128, 50]),
        # NormalizeIntensityd(keys=['image'], nonzero=False),
        EnsureTyped(keys=['image', 'label']),
    ]
)
#
train_ds = CacheDataset(data=train_dicts, transform=train_transform)
val_ds = CacheDataset(data=val_dicts, transform=val_transform)
# train_ds = Dataset(data=train_dicts, transform=train_transform)
# val_ds = Dataset(data=val_dicts, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1, collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available())


# first, check data shape and visualize
'''
data_example = val_ds[2]
print(f"image shape: {data_example['image'].shape}")
print(f"label shape: {data_example['label'].shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title('image')
plt.imshow(data_example['image'][0, :, :, 65].detach().cpu(), cmap="gray")
plt.subplot(1,2,2)
plt.title('label')
plt.imshow(data_example['label'][0, :, :, 65].detach().cpu())
plt.show()
'''

# create train model, loss and optimizer

num_epochs = 400
# num_epochs = 4
val_interval = 2
best_metric = -1
best_metric_epoch = -1
VAL_AMP = True
best_metrics_epochs = [[], []]
epoch_loss_value = list()
metric_values = list()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=2,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm=Norm.BATCH,
#     # dropout=0.2
# ).to(device)

model = ModUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    features=(16, 32, 64, 128, 256),
    kernel_size=(3, 3),
    dilation=(1, 1),
    upsample='deconv',
).to(device)

# model = basicunet(spatial_dims=3, in_channels=1, out_channels=2, features=(32, 32, 64, 128, 256, 32))

# loss_function = DiceLoss(to_onehot_y=True, softmax=True)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

# optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-3)

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

dice_metric = DiceMetric(include_background=True, reduction='mean')
post_trans_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])
post_trans_label = Compose([AsDiscrete(to_onehot=3)])

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            input,
            roi_size=(200, 200, 40),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True


total_start = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    print('-' * 30)
    print(f"epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    step_time_counter = []
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        torch.cuda.empty_cache()
        print(f"step time consuming of epoch {epoch + 1} is: {(time.time() - step_start):.4f}s")
        step_time_counter.append((time.time() - step_start))
    average_step_time = sum(step_time_counter) / step
    epoch_loss /= step
    epoch_loss_value.append(epoch_loss)
    print(step_time_counter)
    print(f"epoch: {epoch+1}, average loss: {epoch_loss:.4f}")
    print(f"average step time consuming of epoch {epoch + 1} is: {average_step_time:.4f}s")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                val_outputs = inference(val_images)
                val_outputs = [post_trans_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_trans_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(data_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")
            print(f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
            print(f"time consuming of epoch {epoch + 1} is: {((time.time() - epoch_start) / 3600):.4f}h")
    else:
        print(f"time consuming of epoch {epoch + 1} is: {((time.time() - epoch_start) / 3600):.4f}h")
total_time = total_start - time.time()
print(f"train completed, best metric is: {best_metric:.4f}, best metric epoch is: {best_metric_epoch}")
print(f"total time consuming : {(total_time / 3600):.4f}h")

# visualization of loss and metric
plt.figure("train process", (16, 8))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_value))]
y = epoch_loss_value
plt.xlabel("epoch")
plt.plot(x, y, color='red')
plt.subplot(1, 2, 2)
plt.title('Val Mean Dice')
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color='green')
plt.show()



# check best pytorch model output with the input image and label
model.load_state_dict(torch.load(os.path.join(data_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    val_image = val_ds[2]['image'].unsqueeze(0).to(device)
    val_output = inference(val_image)
    val_output = torch.argmax(val_output[0], dim=0)
    plt.figure("result", (24, 8))
    plt.subplot(1, 3, 1)
    plt.title("original image")
    plt.imshow(val_ds[2]['image'][0, :, :, 65].detach().cpu(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title('label')
    plt.imshow(val_ds[2]['label'][0, :, :, 65].detach().cpu())
    plt.subplot(1, 3, 3)
    plt.title('output')
    plt.imshow(val_output[:, :, 65].detach().cpu())
    plt.show()






