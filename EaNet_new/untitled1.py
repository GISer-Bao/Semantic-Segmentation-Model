
from src import EaNet
import torch

net = EaNet(in_channel=4, n_classes=2)
net.cuda()
net.train()

for i in range(10):
    #  with torch.no_grad():
    in_ten = torch.randn((1, 4, 492, 492)).cuda()
    logits = net(in_ten)
    print(i)
    print(logits.size())

# print(net.backbone.conv1)
