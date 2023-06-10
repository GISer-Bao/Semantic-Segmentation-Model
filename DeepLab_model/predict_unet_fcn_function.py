import os
import time
import cv2
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import deeplabv3_resnet50, deeplabv3_resnet101
from src_fcn import fcn_resnet50, fcn_resnet101
from src_unet import UNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

'''
函数：预测图像：
img_path 图像所在路径
weights_path 模型权重所在路径
model_name 模型名称 （"fcn_resnet50", "U-net", "deeplab_resnet50", "fcn_resnet101", "deeplab_resnet101"）
'''

def predict_img(img_path: str, weights_path: str, model_name: str):
    
    classes = 1
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))    
    
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    
    # load data
    data_transform = transforms.Compose([transforms.Resize(480),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])
    original_img = Image.open(img_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)
    channel = img.shape[1]
    
    # create model
    if model_name in "fcn_resnet50":
        model = fcn_resnet50(aux=False, num_classes=classes+1)
    elif model_name in "unet":
        model = UNet(in_channels=channel, num_classes=classes+1, base_c=64)
    elif model_name in "deeplab_resnet50":
        model = deeplabv3_resnet50(aux=False, num_classes=classes+1)
    elif model_name in "fcn_resnet101":
        model = fcn_resnet101(aux=False, num_classes=classes+1)
    elif model_name in "deeplab_resnet101":
        model = deeplabv3_resnet101(aux=False, num_classes=classes+1)
        
    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    if model_name in ["fcn_resnet50", "deeplab_resnet50", "fcn_resnet101", "deeplab_resnet101"]:      
        for k in list(weights_dict.keys()):
            if "aux" in k:
                del weights_dict[k]
    model.load_state_dict(weights_dict)
    model.to(device)
    
    # predict
    model.eval()  # 进入验证模式
    with torch.no_grad():
        channel, img_height, img_width = img.shape[-3:]
        init_img = torch.zeros((1, channel, img_height, img_width), device=device)
        model(init_img)
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time ({}): {}".format(model_name, t_end - t_start))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)

    return mask


if __name__ == '__main__':
    
    image_list =[]
    path = r"E:\gaofen-competition\GaoFen_challenge_TVT\test"
    # path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val"
    ids = 801
    example_image = os.path.join(path,'Images\\' + str(ids) + "_img.jpg")
    example_mask = os.path.join(path,'Class\\' + str(ids) + "_img.png")
    
    image = cv2.cvtColor(cv2.imread(example_image), cv2.COLOR_BGR2RGB)
    refer_mask = cv2.cvtColor(cv2.imread(example_mask),cv2.COLOR_BGR2RGB)
    # refer_mask = cv2.cvtColor(cv2.imread(example_mask),cv2.COLOR_BGR2GRAY)
    image_list.append(image)
    image_list.append(refer_mask)

    unet_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\unet_best_model_41.pth"
    mask_unet = predict_img(img_path=example_image, weights_path=unet_weights_path, model_name='unet')
    image_list.append(mask_unet)
    
    ## model
    fcn50_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet50_best_model_46.pth"
    mask_fcn50 = predict_img(img_path=example_image, weights_path=fcn50_weights_path, model_name='fcn_resnet50')
    image_list.append(mask_fcn50)
    
    fcn101_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet101_best_model_50.pth"
    mask_fcn101 = predict_img(img_path=example_image, weights_path=fcn101_weights_path, model_name='fcn_resnet101')    
    image_list.append(mask_fcn101)
    
    deeplab50_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\deeplab_resnet50_best_model_38.pth"
    mask_deeplab50 = predict_img(img_path=example_image, weights_path=deeplab50_weights_path, model_name='deeplab_resnet50')
    image_list.append(mask_deeplab50)

    deeplab101_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\deeplab_resnet101_best_model_31.pth"
    mask_deeplab101 = predict_img(img_path=example_image, weights_path=deeplab101_weights_path, model_name='deeplab_resnet101')
    image_list.append(mask_deeplab101)

    
    # plot the image, mask and multiplied together
    plt.figure(figsize=(40, 30))
    # plt.figure(figsize=(22, 10))
    title = ['Raw image','Ground truth','U-net','FCN-Resnet50','FCN-Resnet101','DeepLabV3-Resnet50','DeepLabV3-Resnet101']
    for i in range(len(image_list)):
        # plt.subplot(1, len(image_list), i+1)
        plt.subplot(1, 7, i+1)
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.01,hspace=0.2)     
        if i > 1:
            temp = np.expand_dims(image_list[i],axis=2).repeat(3,axis=2) * 255
            plt.imshow(temp)
            plt.axis('on')
            plt.xticks([])
            plt.yticks([])
            plt.title(title[i],fontsize=24)
        else:
            plt.imshow(image_list[i])
            plt.axis('on')
            plt.xticks([])
            plt.yticks([])
            plt.title(title[i],fontsize=24)
  
    # # ax3.imshow(refer_mask,cmap='gray')
    # ax2.imshow(refer_mask)
    # ax2.axis('off')
    # ax2.set_title('Ground truth',fontsize=26)
    






