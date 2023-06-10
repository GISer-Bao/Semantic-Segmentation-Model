import os
import torch

from src import fcn_resnet50,fcn_resnet101
from train_utils import evaluate
from MyDataset_v2 import my_dataset, read_split_data
import albumentations as A


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # transforms
    test_img_aug = A.Compose([A.RandomCrop(width=480, height=480),])

    train_images_path, train_masks_path, test_images_path, test_masks_path = read_split_data(args.data_path,val_rate=1)

    test_dataset = my_dataset(images_path=test_images_path,
                               masks_path=test_masks_path,
                               transforms=None)  
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             drop_last=False,
                                             pin_memory=True)
    
    model = fcn_resnet101(aux=args.aux, num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat, dice = evaluate(model, test_loader, device=device, num_classes=num_classes)
    print(str(confmat))
    print(f"dice coefficient: {dice*100:.3f}\n")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 validation")

    data_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\test"
    parser.add_argument("--data-path", default=data_path, help="test dataset root")
    weights_path = r"E:\gaofen-competition\GaoFen_challenge_TVT\train_val\save_weights\fcn_resnet101_best_model_50.pth"
    parser.add_argument("--weights", default=weights_path)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
