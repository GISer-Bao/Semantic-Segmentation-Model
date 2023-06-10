

import os
import datetime
import torch

from torch.utils import data
import transforms as T
from my_dataset import DUTSDataset

import torch.nn.functional as F
from src import EaNet
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
import albumentations as A
from MyDataset_v2 import my_dataset, read_split_data

name = 'EaNet'
batch_size = 2 
device = 'cuda'
device = torch.device(device if torch.cuda.is_available() else "cpu")

# 用来保存训练以及验证过程中信息

# transforms
train_img_aug = A.Compose([
    A.Resize(width=320, height=320),
    A.RandomCrop(width=288, height=288),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(limit=180, p=0.5),
])
val_img_aug = A.Compose([A.Resize(width=320, height=320),])

data_path = r"E:\GID_experiment\allocate_data_492\train_u2net_ealoss"
train_images_path, train_masks_path, val_images_path, val_masks_path = read_split_data(
    root=data_path,images_format='.tif',masks_format='.tif', val_rate=0.25)
train_dataset = my_dataset(images_path=train_images_path,
                           masks_path=train_masks_path,
                           transforms=train_img_aug)
val_dataset = my_dataset(images_path=val_images_path,
                           masks_path=val_masks_path,
                           transforms=None)  

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=0,
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=True)

val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=1,
                                         num_workers=0,
                                         drop_last=False,
                                         pin_memory=True)

model = EaNet(in_channel=4, n_classes=2).to(device)

# image = list(enumerate(val_data_loader))[3][1][0].cuda()
# label = list(enumerate(val_data_loader))[3][1][1].cuda()
# print(image.shape)
# print(label.shape)

# pred_img = model(image)
# print(len(pred_img))
# print(pred_img.shape)

from train_utils import criterion

model.train()
for image, target in val_data_loader:
    image, target = image.to(device), target.to(device)
    with torch.cuda.amp.autocast(enabled=True):
        output = model(image)
        target = target.squeeze(1).to(torch.long)
        print(output.shape)
        print(target.shape)
        loss = criterion(output, target, n_min=1*492*492//16, num_classes=2, ignore_index=255)
        print(loss)

# model.eval()
# for image, target in val_data_loader:
#     image, target = image.to(device), target.to(device)
#     with torch.cuda.amp.autocast(enabled=True):
#         output = model(image)
#         target = target.squeeze(1).to(torch.long)
#         print(output.shape)
#         print(target.shape)
#         loss = criterion(output, target, n_min=1*492*492//16, num_classes=2, ignore_index=255)
#         print(loss)






