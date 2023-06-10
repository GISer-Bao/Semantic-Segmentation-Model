
import os
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as T
import albumentations as A


'''
函数：将数据集划分为训练集和验证集：

root 输入文件所在目录
val_rate 验证数据集的占比，训练数据集为 "1-val_rate"
images_format 图像的文件后缀
masks_format 标签的文件后缀

'''


def read_split_data(root: str, val_rate: float = 0.2, images_format: str='.jpg',masks_format: str='.png'):
    random.seed(0)  # 保证随机结果可复现    
    assert os.path.exists(root), "dataset root path: {} does not exist.".format(root)
   
    images_dir = os.path.join(root, 'Images')
    assert os.path.exists(root), "Images path '{}' does not exist.".format(root)  
    masks_dir = os.path.join(root, 'Masks')
    assert os.path.exists(root), "Masks path '{}' does not exist.".format(root)

    # supported = [".jpg", ".png"]  # 支持的文件后缀类型
    files_name = [os.path.splitext(i)[0] for i in os.listdir(images_dir)
                  if os.path.splitext(i)[-1] in images_format]
    val_files_name = random.sample(files_name, k=int(len(files_name) * val_rate))
    train_files_name = [file_name for file_name in files_name
                         if file_name not in val_files_name]
    
    train_images_path = [os.path.join(images_dir, name + images_format) for name in train_files_name]  # 存储训练集的所有图片路径
    train_images_label = [os.path.join(masks_dir, name + masks_format) for name in train_files_name]  # 存储训练集图片对应索引信息
    val_images_path = [os.path.join(images_dir, name + images_format) for name in val_files_name]  # 存储验证集的所有图片路径
    val_images_label = [os.path.join(masks_dir, name + masks_format) for name in val_files_name]  # 存储验证集图片对应索引信息

    print("{} images for the dataset.".format(len(files_name)))
    print("{} images for training.".format(len(train_files_name)))
    print("{} images for validation.".format(len(val_files_name)))

    return train_images_path, train_images_label, val_images_path, val_images_label

'''
类：处理图像和标签：

images_path 图像文件路径
masks_path _rate 标签文件路径
transforms 图像和标签的变换

'''

class my_dataset(data.Dataset):
    def __init__(self, images_path: list, masks_path: list, transforms=None):
        super(my_dataset, self).__init__()
        self.images_path = images_path
        self.images_label = masks_path
        self.transforms = transforms
        self.img_as_tensor3 = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
            ])
        self.img_as_tensor4 = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.496, 0.370, 0.390, 0.362], 
                std=[0.241, 0.229, 0.222, 0.219]),
            ])
        self.mask_as_tensor = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label) where label is the image segmentation.
        """
        # img = Image.open(self.images_path[index]).convert('RGB')
        # label = Image.open(self.images_label[index]).convert('L')
        
        img = np.array(Image.open(self.images_path[index]))
        target = Image.open(self.images_label[index])
        # target = np.expand_dims(np.array(target),axis=0) / 255
        target = np.array(target) / 255
        # target = np.array(target, dtype=np.uint8)
        
        if self.transforms is not None:
            augments = self.transforms(image=img,mask=target)
            img, target = augments['image'], augments['mask']
        
        # np.array -> torch.tensor
        if img.shape[2]==4:
            img = self.img_as_tensor4(img)
        elif img.shape[2]==3:
            img = self.img_as_tensor3(img)
        target = self.mask_as_tensor(target)
        # target = torch.as_tensor(target, dtype=torch.int64)
        # target = T.functional.to_tensor(target)
        return img, target
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == '__main__':
    
    # transforms
    train_img_aug = A.Compose([
        A.RandomCrop(width=400, height=400),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Flip(p=0.5),
        A.RandomGridShuffle(grid=(2, 2), p=0.5),
        A.Rotate(limit=180, p=0.5),
    ])
    val_img_aug = A.Compose([A.RandomCrop(width=480, height=480),])
    
    data_path = r"E:\GaoFen_challenge\GaoFen_challenge_TrVa"
    train_images_path, train_masks_path, val_images_path, val_masks_path = read_split_data(data_path,val_rate=0.25)
    dataset = my_dataset(images_path=train_images_path, masks_path=train_masks_path, transforms=None)
    
    d1 = dataset[5]
    
    print(d1[0].shape)
    print(d1[0].dtype)
    print(d1[1].shape)
    print(d1[1].dtype)

