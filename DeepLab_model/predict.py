import os
import time
import cv2
import matplotlib.pyplot as plt
import glob
import tqdm
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import deeplabv3_resnet50, deeplabv3_resnet101


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


'''
函数：预测图像：
img_path 图像所在路径
weights_path 模型权重所在路径

'''

def main(img_path: str, weights_path: str):
    
    classes = 1
    weights_path = weights_path
    img_path = img_path
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    

    

    H, W = np.array(cv2.imread(img_path,-1)).shape[0], np.array(cv2.imread(img_path,-1)).shape[0]
    # create model
    model = deeplabv3_resnet101(aux=False, num_classes=classes+1)
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]
    model.load_state_dict(weights_dict)
    model.to(device)

    # load data
    data_transform = transforms.Compose([transforms.Resize(480),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.496, 0.370, 0.390, 0.362),
                                                              std=(0.241, 0.229, 0.222, 0.219))])

    original_img = Image.open(img_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)
    
    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 4, img_height, img_width), device=device)
        model(init_img)
        output = model(img.to(device))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = cv2.resize(prediction, (H, W), interpolation=cv2.INTER_NEAREST)
        return mask

    # predict
    model.eval()  # 进入验证模式
    with torch.no_grad():
        channel, img_height, img_width = img.shape[-3:]
        init_img = torch.zeros((1, channel, img_height, img_width), device=device)
        model(init_img)
        output = model(img.to(device))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = cv2.resize(prediction, (H, W), interpolation=cv2.INTER_NEAREST)

    return mask


if __name__ == '__main__':
    print('\n-------------Processing Start!!!-------------\n')
    
    # weights_path = r"E:\GID\exp\train_val\save_weights_old\unet_best_model_13.pth"
    weights_path = r"E:\GID_experiment\allocate_data_492\train_deepval\save_weights_deepval\deeplab_resnet101_best_model_epoch21_dice0.7511841654777527.pth"
    imgs_path = r"E:\GID_experiment\allocate_data_492\test\image_NirRGB"
    masks_path = os.path.dirname(imgs_path) + "\\test_result_deeplab-resnet101_epoch21"
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
        
    imgs_name_list = glob.glob(imgs_path + '\\*.tif')
    
    start_time = time.time()
    for img in tqdm.tqdm(imgs_name_list, desc='Precessing'):
        mask = main(img,weights_path)
        img_name = os.path.basename(img)
        cv2.imwrite(os.path.join(masks_path,img_name.split('.')[0]+".tif"), mask*255)
        
    print('\n\n-------------Congratulations! Processing Done!!!-------------')
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))
    







