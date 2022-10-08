import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as trans
import torch.utils.data as data
import PIL.Image as pimg
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#28*28*128

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1,groups=4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )#14*14*256

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1,groups=4),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )#7*7*512

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=7*7*512,out_features=10),
        )#10

    def forward(self,x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        # y3 = torch.reshape(y3,[-1,7*7*512])
        y3 = torch.reshape(y3,[x.size(0),-1])
        self.y4 = self.layer4(y3)
        # out = F.softmax(self.y4)
        out = self.y4
        return out
if __name__ == '__main__':
    transf_data = trans.Compose(
        [trans.ToTensor(),
        trans.Normalize(mean=[0.5,],std=[0.5,])]
    )

    test_data = datasets.MNIST("../data",train=False,transform=transf_data,download=False)
    test_loader = data.DataLoader(test_data,100,shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = torch.load("./model.pt").to(device)# 调用模型
    loss_function = nn.MSELoss()
    # loss_function = nn.CrossEntropyLoss()

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i,(x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        ys = torch.zeros(y.cpu().size(0),10).scatter_(1,y.cpu().reshape(-1,1),1).cuda()
        # loss = loss_function(net.y6,ys)
        loss = loss_function(out,ys)
        print("Test_Loss:{:.3f}".format(loss.item()))
        eval_loss += loss.item()*y.size(0)

        arg_max= torch.argmax(out,1)
        eval_acc += (arg_max==y).sum().item()

    mean_loss = eval_loss/len(test_data)
    mean_acc= eval_acc/len(test_data)
    print("Loss:{:.3f},Acc:{:.3f}".format(mean_loss,mean_acc))



