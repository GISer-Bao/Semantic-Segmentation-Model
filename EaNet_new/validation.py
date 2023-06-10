import os
from typing import Union, List
import torch

from torch.utils import data
from my_dataset import DUTSDataset
import albumentations as A

from src import u2net_full
from train_utils import evaluate
import transforms as T
from MyDataset_v2 import my_dataset, read_split_data



class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # test_img_aug = A.Compose([A.RandomCrop(width=480, height=480),])

    train_images_path, train_masks_path, test_images_path, test_masks_path = read_split_data(args.data_path,val_rate=1)

    test_dataset = my_dataset(images_path=test_images_path,
                               masks_path=test_masks_path,
                               transforms=None)  

    val_data_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             drop_last=False,
                                             pin_memory=True)

    model = u2net_full()
    pretrain_weights = torch.load(args.weights, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
    print(mae_metric, f1_metric)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net validation")

    data_path = r"E:\gaofen-competition\GaoFen_challenge_Te"
    parser.add_argument("--data-path", default = data_path, help="DUTS root")
    weight_path = r"E:\gaofen-competition\GaoFen_challenge_TrVa\save_weights_u2net\u2net_best_model_epoch70.pth"
    parser.add_argument("--weights", default=weight_path)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
