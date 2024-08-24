import torch
import os
import glob
import tqdm
import matplotlib.pyplot as plt
from monai.data import Dataset, CacheDataset
from monai.data import DataLoader, decollate_batch, pad_list_data_collate
from monai.config import print_config
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import basicunet, UNet
from monai.networks.nets.modunet import ModUNet
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
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
    SaveImaged,
    EnsureTyped,
    EnsureChannelFirstd,
)

set_determinism(seed=0)

data_dir = r"/home/kevin/anaconda3/envs/pytorch/Task03_Liver"
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
train_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[:50], train_labels[:50])
]
val_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images[50:81], train_labels[50:81])

]
print(train_dicts)
print(val_dicts)
print(val_dicts[2])

# define transform for image and segmentation
train_transform = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=[1.35, 1.35, 1.35], mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        NormalizeIntensityd(keys=['image'], nonzero=False),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        RandCropByPosNegLabeld(keys=['image','label'], label_key='label', spatial_size=[168, 168, 50], pos=2, neg=1, num_samples=5),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
        EnsureTyped(keys=['image', 'label']),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=[1.35, 1.35, 1.35], mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        NormalizeIntensityd(keys=['image'], nonzero=False),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        EnsureTyped(keys=['image', 'label']),
    ]
)
# train_ds = Dataset(data=train_dicts, transform=train_transform)
# val_ds = Dataset(data=val_dicts, transform=val_transform)
train_ds = CacheDataset(data=train_dicts, transform=train_transform)
val_ds = CacheDataset(data=val_dicts, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1, collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=pad_list_data_collate, pin_memory=torch.cuda.is_available())

# first, check data shape and visualize

data_example = val_ds[2]
print(val_ds[2]['image'].shape)
print(f"image shape: {data_example['image'].shape}")
print(f"label shape: {data_example['label'].shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title('image')
plt.imshow(data_example['image'][0, :, :, 55].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title('label')
plt.imshow(data_example['label'][0, :, :, 55].detach().cpu())
plt.show()

# create train model, loss and optimizer

num_epochs = 400
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
    features=(16, 32, 64, 128),
    kernel_size=(3, 3),     # Two Convolution for each layer. e.g. kernel_size=(3, 5) 1. used for 3*3, 2. used for 5*5
    dilation=(1, 1),        # # Two Convolution for each layer. e.g. dilation=(1, 2) 1. used for 1 and 2. used for 2
    upsample='deconv',
).to(device)


loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
dice_metric = DiceMetric(include_background=True, reduction='mean')
post_trans_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])
post_trans_label = Compose([AsDiscrete(to_onehot=3)])
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            input,
            roi_size=(168, 168, 50),
            sw_batch_size=2,
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

for epoch in range(num_epochs):
    print('-' * 30)
    print(f"epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # outputs = model(inputs)
        # loss = loss_function(outputs, labels)
        # loss.backward()
        # optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        torch.cuda.empty_cache()
    epoch_loss /= step
    epoch_loss_value.append(epoch_loss)
    print(f"epoch: {epoch+1}, average loss: {epoch_loss:.4f}")

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
print(f"train completed, best metric is: {best_metric:.4f}, best metric epoch is: {best_metric_epoch}")

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
    val_image = val_ds[4]['image'].unsqueeze(0).to(device)
    val_output = inference(val_image)
    val_output = torch.argmax(val_output[0], dim=0)
    plt.figure("result", (24, 8))
    plt.subplot(1, 3, 1)
    plt.title("original image")
    plt.imshow(val_ds[2]['image'][0, :, :, 55].detach().cpu(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title('label')
    plt.imshow(val_ds[2]['label'][0, :, :, 55].detach().cpu())
    plt.subplot(1, 3, 3)
    plt.title('output')
    plt.imshow(val_output[:, :, 55].detach().cpu())
    plt.show()





