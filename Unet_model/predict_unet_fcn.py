import os
import time
import cv2
import matplotlib.pyplot as plt


import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet
from src_fcn import fcn_resnet50

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main_unet(img_path: str,weights_path: str):
    classes = 1
    weights_path = weights_path
    img_path = img_path
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=64)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(480),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    original_img = Image.open(img_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)
        return mask

        # 将前景对应的像素值改成255(白色)
        # prediction[prediction == 1] = 255
        # mask = Image.fromarray(prediction)
        # mask.save("test_result.png")
        
def main_fcn(img_path: str,weights_path: str):
    classes = 1
    weights_path = weights_path
    img_path = img_path
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=False, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(480),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    original_img = Image.open(img_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = cv2.resize(prediction, (492, 492), interpolation=cv2.INTER_NEAREST)
        return mask

if __name__ == '__main__':
    
    unet_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\unet_best_model_41.pth"
    fcn_weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet50_best_model_46.pth"
    path = r"E:\gaofen-competition\GaoFen_challenge_TVT\test"
    # path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val"
    
    ids = 123
    example_image = os.path.join(path,'Images\\' + str(ids) + "_img.jpg")
    example_mask = os.path.join(path,'Class\\' + str(ids) + "_img.png")
 
    unet_mask = main_unet(example_image,unet_weights_path)
    fcn_mask = main_fcn(example_image,fcn_weights_path)
    
    image = cv2.cvtColor(cv2.imread(example_image), cv2.COLOR_BGR2RGB)
    refer_mask = cv2.cvtColor(cv2.imread(example_mask),cv2.COLOR_BGR2GRAY)

    #plot the image, mask and multiplied together
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20, 15))

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Raw image',fontsize=26)
    
    # ax3.imshow(refer_mask,cmap='gray')
    ax2.imshow(refer_mask)
    ax2.axis('off')
    ax2.set_title('Ground truth',fontsize=26)
    
    ax3.imshow(unet_mask)
    ax3.axis('off')
    ax3.set_title('U-net',fontsize=26)

    # ax3.imshow(refer_mask,cmap='gray')
    ax4.imshow(fcn_mask)
    ax4.axis('off')
    ax4.set_title('FCN-ResNet50',fontsize=26)




