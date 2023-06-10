
import os
import numpy as np
import matplotlib.pyplot as plt

def get_acc(path: str):
    # 读取文件
    lines = open(path,'r').readlines()
    # 删除 "\n"
    info = [line.strip('\n') for line in lines if line.strip('\n') != '']
    
    epoch = [ float(i.split(' ')[1][:-1]) for i in info]
    loss = [ float(i.split(' ')[3]) for i in info]
    mae = [ float(i.split(' ')[7]) for i in info]
    maxF1 = [ float(i.split(' ')[9]) for i in info]
    
    return epoch, loss, mae, maxF1

def save_plots(epoch, loss, mae, maxF1, acc_name: str):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # loss plots
    plt.figure(figsize=(10, 7))
    
    plt.plot(epoch, loss, color='red', linestyle='-', marker='v', label='loss')
    plt.plot(epoch, mae, color='blue', linestyle=':', marker='o', label='MAE')
    plt.plot(epoch, maxF1, color='cyan', linestyle=':', marker='s', label='maxF1')
    
    plt.tick_params(labelsize=18)
    # plt.xticks(np.arange(0,300,20))
    plt.yticks(np.arange(0,1,1))
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel(acc_name, fontsize=20)
    plt.title(acc_name, fontsize=26)
    plt.legend(prop={'size':20})
    # plt.savefig(os.path.join(data_dir, acc_name+".png"))

if __name__ == '__main__':
    
    acc_dir = r"E:\gaofen-competition\GaoFen_challenge_TrVa\save_weights_u2net\u2net_20220831-222721.txt"
    epoch, loss, mae, maxF1 = get_acc(path = acc_dir)
    save_plots(epoch=epoch, loss=loss, mae=mae, maxF1=maxF1, acc_name = 'accuracy')
