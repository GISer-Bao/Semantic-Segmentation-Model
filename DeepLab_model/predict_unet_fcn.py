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

def main(img_path: str,weights_path_fcn50: str,weights_path_unet: str,weights_path_deeplab50: str,
         weights_path_fcn101: str,weights_path_deeplab101: str):
    classes = 1
    
    weights_path_fcn50 = weights_path_fcn50
    weights_path_unet = weights_path_unet
    weights_path_deeplab50 = weights_path_deeplab50
    # weights_path_fcn101 = weights_path_fcn101
    # weights_path_deeplab101 = weights_path_deeplab101
    
    img_path = img_path
    
    assert os.path.exists(weights_path_fcn50), f"weights {weights_path_fcn50} not found."
    assert os.path.exists(weights_path_unet), f"weights {weights_path_unet} not found."
    assert os.path.exists(weights_path_deeplab50), f"weights {weights_path_deeplab50} not found."
    # assert os.path.exists(weights_path_fcn101), f"weights {weights_path_fcn101} not found."
    # assert os.path.exists(weights_path_deeplab101), f"weights {weights_path_deeplab101} not found."
    
    assert os.path.exists(img_path), f"image {img_path} not found."
    
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model_fcn50 = fcn_resnet50(aux=False, num_classes=classes+1)
    model_unet = UNet(in_channels=3, num_classes=classes+1, base_c=64)
    model_deeplab50 = deeplabv3_resnet50(aux=False, num_classes=classes+1)
    model_fcn101 = fcn_resnet101(aux=False, num_classes=classes+1)
    model_deeplab101 = deeplabv3_resnet101(aux=False, num_classes=classes+1)


    # load weights,fcn_resnet50
    weights_dict_fcn50 = torch.load(weights_path_fcn50, map_location='cpu')['model']
    for k in list(weights_dict_fcn50.keys()):
        if "aux" in k:
            del weights_dict_fcn50[k]
    model_fcn50.load_state_dict(weights_dict_fcn50)
    model_fcn50.to(device)    
    # load weights,unet
    model_unet.load_state_dict(torch.load(weights_path_unet, map_location='cpu')['model'])
    model_unet.to(device)
    # load weights,deeplab_resnet50
    weights_dict_deeplab50 = torch.load(weights_path_deeplab50, map_location='cpu')['model']
    for k in list(weights_dict_deeplab50.keys()):
        if "aux" in k:
            del weights_dict_deeplab50[k]
    model_deeplab50.load_state_dict(weights_dict_deeplab50)
    model_deeplab50.to(device)
    # load weights,fcn_resnet101
    weights_dict_fcn101 = torch.load(weights_path_fcn101, map_location='cpu')['model']
    for k in list(weights_dict_fcn101.keys()):
        if "aux" in k:
            del weights_dict_fcn101[k]
    model_fcn101.load_state_dict(weights_dict_fcn101)
    model_fcn101.to(device)    
    # load weights,deeplab_resnet101
    weights_dict_deeplab101 = torch.load(weights_path_deeplab101, map_location='cpu')['model']
    for k in list(weights_dict_deeplab101.keys()):
        if "aux" in k:
            del weights_dict_deeplab101[k]
    model_deeplab101.load_state_dict(weights_dict_deeplab101)
    model_deeplab101.to(device)



    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(480),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    original_img = Image.open(img_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    # predict, fcn50
    model_fcn50.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model_fcn50(init_img)
        t_start = time_synchronized()
        output = model_fcn50(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time (fcn_resnet50): {}".format(t_end - t_start))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask_fcn50 = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)
    # predict, unet
    model_unet.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model_unet(init_img)
        t_start = time_synchronized()
        output = model_unet(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time (unet): {}".format(t_end - t_start))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask_unet = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)
    # predict, deeplab50
    model_deeplab50.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model_deeplab50(init_img)
        t_start = time_synchronized()
        output = model_deeplab50(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time (deeplab_resnet50): {}".format(t_end - t_start))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask_deeplab50 = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)
    # predict, fcn101
    model_fcn101.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model_fcn101(init_img)
        t_start = time_synchronized()
        output = model_fcn101(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time (fcn_resnet101): {}".format(t_end - t_start))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask_fcn101 = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)
    # predict, deeplab101
    model_deeplab101.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model_deeplab101(init_img)
        t_start = time_synchronized()
        output = model_deeplab101(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time (deeplab_resnet101): {}".format(t_end - t_start))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask_deeplab101 = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)    

    
    return mask_fcn50, mask_unet, mask_deeplab50, mask_fcn101, mask_deeplab101



if __name__ == '__main__':
    image_list =[]
    unet_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\unet_best_model_41.pth"
    fcn50_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet50_best_model_46.pth"
    deeplab50_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\deeplab_resnet50_best_model_38.pth"
    fcn101_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet101_best_model_50.pth"
    deeplab101_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\deeplab_resnet101_best_model_31.pth"
    
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

    mask_fcn50, mask_unet, mask_deeplab50, mask_fcn101, mask_deeplab101 = main(
        example_image, fcn50_weights_path, unet_weights_path, 
        deeplab50_weights_path, fcn101_weights_path, deeplab101_weights_path)
    image_list.append(mask_unet)
    image_list.append(mask_fcn50)
    image_list.append(mask_fcn101)
    image_list.append(mask_deeplab50)
    image_list.append(mask_deeplab101)
    
    # image_list.append(mask_deeplab50)
    
    #plot the image, mask and multiplied together
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
    






