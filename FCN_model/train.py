import os
import time
import datetime

import torch

from src import fcn_resnet50,fcn_resnet101
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from MyDataset_v2 import my_dataset, read_split_data
import albumentations as A


def main(args):
    name = 'fcn_resnet101'
    batch_size = args.batch_size
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = model_path + "\\" + name + "_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # transforms
    size = 480
    train_img_aug = A.Compose([
        A.Resize(width=size, height=size),
        A.RandomCrop(width=size, height=size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Flip(p=0.5),
        A.RandomGridShuffle(grid=(2, 2), p=0.5),
        A.Rotate(limit=180, p=0.5),
    ])
    val_img_aug = A.Compose([A.Resize(width=size, height=size),])

    train_images_path, train_masks_path, val_images_path, val_masks_path = read_split_data(
        args.data_path,val_rate=0.25, images_format='.tif',masks_format='.tif')

    train_dataset = my_dataset(images_path=train_images_path,
                               masks_path=train_masks_path,
                               transforms=train_img_aug)
    val_dataset = my_dataset(images_path=val_images_path,
                               masks_path=val_masks_path,
                               transforms=None)  
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             drop_last=False,
                                             pin_memory=True)
    # model 需要修改
    model = fcn_resnet101(aux=args.aux, num_classes=num_classes)
    model.to(device)
    # optimizer 需要修改
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)


    best_dice = 0.
    start_time = time.time()
    for epoch in range(0, args.epochs):
        start_time_epoch = time.time()
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice*100:.3f}\n")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch+1}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice*100:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        
        if args.save_best is True:
            torch.save(save_file, model_path + "\\"+name+"_best_model_epoch{}_dice{}.pth".format(epoch+1,dice))
        elif epoch % 20 == 0:
            torch.save(save_file, model_path + "\\"+name+"_model_epoch{}_dice{}.pth".format(epoch+1,dice))
        elif epoch == args.epochs - 1:
            torch.save(save_file, model_path + "\\"+name+"_model_epoch{}_dice{}.pth".format(epoch+1,dice))

        epoch_time = time.time() - start_time_epoch
        print("Epoch {}/{} : {:.0f}m {:.2f}s\n".format(epoch+1, args.epochs,
                                                     epoch_time // 60, epoch_time % 60))

    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch model training")

    data_path = r"./train_val"
    parser.add_argument("--data-path", default=data_path, help="Dataset root")
    model_dir = r"./save_weights_fcn"
    parser.add_argument("--model-path", default=model_dir, help="Saved model root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=3, type=int)
    parser.add_argument("--epochs", default=180, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')

    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    args.data_path = r"E:\GID_experiment\allocate_data_492\train_unet"
    args.model_path = r"E:\GID_experiment\allocate_data_492\train_fcn\save_weights_fcn"    
    args.in_ch = 4
    main(args)

