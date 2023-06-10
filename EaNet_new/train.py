
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import datetime
import torch

# from src import u2net_full
from src import EaNet
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
import albumentations as A
from MyDataset_v2 import my_dataset, read_split_data


def main(args):
    name = 'EaNet'
    batch_size = args.batch_size  
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    size = 480
    train_n_min = batch_size*size*size//16
    eval_n_min = size*size//16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 用来保存训练以及验证过程中信息
    results_file = model_path + os.sep + name + "_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # transforms
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
        root=args.data_path,images_format='.tif',masks_format='.tif', val_rate=0.25)
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


    model = EaNet(in_channel=args.in_ch, n_classes=2)
    model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # ------- training process --------
    print('\n-------------Start training!-------------\n')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_ealoss, current_mae, current_f1 = 10.0, 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time_epoch = time.time()
        ealoss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, lr_scheduler=lr_scheduler, 
                                     n_min=train_n_min, num_classes=2, print_freq=args.print_freq, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            ealoss, mae_metric, f1_metric = evaluate(model, val_data_loader, device=device, n_min=eval_n_min, num_classes=2)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            print(f"[epoch: {epoch+1}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch+1}] train_loss: {ealoss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} \n"
                f.write(write_info)

            # save_best
            if ealoss <= current_ealoss:
                current_ealoss = ealoss
                torch.save(save_file,model_path +os.sep+name+"_best_model_epoch{}_ealoss{}_mae{}_f1{}.pth".format(
                    epoch+1,ealoss,mae_info,f1_info))
            elif epoch % 20 == 0:
                torch.save(save_file,model_path +os.sep+name+"_model_epoch{}_ealoss{}_mae{}_f1{}.pth".format(
                    epoch+1,ealoss,mae_info,f1_info))
            elif epoch == args.epochs - 1:
                torch.save(save_file,model_path +os.sep+name+"_last_model_epoch{}_ealoss{}_mae{}_f1{}.pth".format(
                    epoch+1,ealoss,mae_info,f1_info))
                    
        epoch_time = time.time() - start_time_epoch
        print("Epoch {}/{} : {:.0f}m {:.2f}s\n".format(epoch+1, args.epochs, epoch_time // 60, epoch_time % 60))
    
    print('\n\n------Congratulations! Training Done!------')
    total_time = time.time() - start_time
    print("Total training time : {:.0f}h {:.0f}m {:.2f}s\n".format(
        total_time//3600, (total_time%3600)//60, (total_time%3600)%60))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch eanet training")

    data_path = r"./"
    parser.add_argument("--data-path", default=data_path, help="Dataset root")
    model_path = r"./save_weights_u2net"
    parser.add_argument("--model-path", default=model_path, help="Saved model root")
    
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--in-ch", default=3, help="input channel")
    parser.add_argument("-b", "--batch-size", default=3, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=360, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=5, type=int, help="validation interval default 5 Epochs")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = parse_args()
    args.data_path = r"E:\GID_experiment\allocate_data_492\train_u2net_ealoss"
    args.model_path = r"E:\GID_experiment\allocate_data_492\train_u2net_ealoss\save_weights_eanet"    
    
    # args.data_path = r"E:\gaofen-competition\GaoFen_challenge_TrVa\train_validation"
    # args.model_path = r"E:\gaofen-competition\GaoFen_challenge_TrVa\save_weights_u2net_ealoss_allside"

    # args.data_path = r"E:\GaoFen_challenge\GaoFen_challenge_TrVa"
    # args.model_path = r"E:\GaoFen_challenge\save_weights_u2net_ealoss"
       
    args.in_ch = 4
    main(args)
