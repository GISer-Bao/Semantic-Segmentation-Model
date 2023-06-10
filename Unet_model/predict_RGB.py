
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
from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict(img_path: str,weights_path: str):

    classes = 1
    channels = 3
    device = 'cpu'
    weights_path = weights_path
    img_path = img_path
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    
    # get devices
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    H, W = np.array(cv2.imread(img_path,-1)).shape[0], np.array(cv2.imread(img_path,-1)).shape[0]
    # create model
    model = UNet(in_channels=channels, num_classes=classes+1, base_c=64)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(480),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    # data_transform = transforms.Compose([transforms.Resize(480),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize(mean=(0.496, 0.370, 0.390, 0.362),
    #                                                           std=(0.241, 0.229, 0.222, 0.219))])
    
    
    original_img = Image.open(img_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, channels, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        # print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = cv2.resize(prediction, (H, W), interpolation=cv2.INTER_NEAREST)

        return mask
    

if __name__ == '__main__':
    print('\n-------------Processing Start!!!-------------\n')
    
    weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\unet_best_model_41.pth"
    # weights_path = r"E:\GID\exp\train_val\save_weights\unet_best_model_11.pth"
    imgs_path = r"E:\GID\exp_rgb\seg\Images"
    masks_path = os.path.dirname(imgs_path) + '\\test_result_unet'
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
        
    imgs_name_list = glob.glob(imgs_path + '\\*.tif')
    
    start_time = time.time()
    for img in tqdm.tqdm(imgs_name_list, desc='Precessing'):
        mask = predict(img,weights_path)
        img_name = os.path.basename(img)
        cv2.imwrite(os.path.join(masks_path,img_name.split('.')[0]+".tif"), mask*255)
        
    print('\n\n-------------Congratulations! Processing Done!!!-------------')
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))

# if __name__ == '__main__':
    
#     weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\Unet_best_model.pth"
#     path = r"E:\gaofen-competition\GaoFen_challenge_TVT\test"
#     # path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val"
    
#     ids = 540
#     example_image = os.path.join(path,'Images\\' + str(ids) + "_img.jpg")
#     example_mask = os.path.join(path,'Class\\' + str(ids) + "_img.png")
 
#     mask = predict(example_image,weights_path)
#     image = cv2.cvtColor(cv2.imread(example_image), cv2.COLOR_BGR2RGB)
#     refer_mask = cv2.cvtColor(cv2.imread(example_mask),cv2.COLOR_BGR2GRAY)

#     #plot the image, mask and multiplied together
#     fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 15))

#     ax1.imshow(image)
#     ax1.axis('off')
#     ax1.set_title('Raw image',fontsize=26)

#     ax2.imshow(mask)
#     ax2.axis('off')
#     ax2.set_title('U-net',fontsize=26)

#     # ax3.imshow(refer_mask,cmap='gray')
#     ax3.imshow(refer_mask)
#     ax3.axis('off')
#     ax3.set_title('Ground truth',fontsize=26)

