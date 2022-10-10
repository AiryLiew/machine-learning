import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #序列构造器内的程序会自上而下的运行
        self.fc1=torch.nn.Sequential(
        torch.nn.Linear(784,256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU())

        self.fc2=torch.nn.Sequential(
        torch.nn.Linear(256,64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU())

        self.fc3=torch.nn.Linear(64,10)

    def forward(self,x):
        # N,C,H,W-->N,V
        x=torch.reshape(x,[x.size(0),-1])
        #X*W
        y1=self.fc1(x)#[N,784]*[784,256]-->[N,256]
        y2=self.fc2(y1)#[N,256]*[256,64]-->[N,64]
        y3=self.fc3(y2)#[N,64]*[64,10]-->[N,10]
        return y3

if __name__ == '__main__':
    # net=Net()
    # print(net)
    # data=torch.randn([10,1,28,28])
    # output1=net(data)
    # output2=net.forward(data)
    # print(output1.shape)
    # print(output2.shape)
    #数据初始化方法
    transf=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5,],std=[0.5,])]
    )
    #下载数据
    train_data=datasets.MNIST("./data",train=True,transform=transf,download=True)
    test_data=datasets.MNIST("./data",train=False,transform=transf,download=False)
    print(train_data.data.shape)
    print(train_data.targets.shape)
    print(train_data.classes)
    print(test_data.data.shape)

    #加载数据
    train_loader=DataLoader(train_data,100,True)
    test_loader=DataLoader(test_data,100,True)

    #指定设备
    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    #实例化网络到具体设备
    net=Net().to(device)
    #定义损失函数
    loss_fn=torch.nn.CrossEntropyLoss()
    #定义优化器
    optim=torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(1):
        for i,(x,y) in enumerate(train_loader):
            x=x.to(device)
            y=y.to(device)
            out=net(x)#前向传播
            loss=loss_fn(out,y)#计算损失
            #梯度下降三部曲，反向传播
            optim.zero_grad()#情况之前的梯度
            loss.backward()#重新计算当前的梯度
            optim.step()#沿着当前的梯度更新一步

            if i%50==0:
                print("loss",loss.item())

    #评估模式，做测试，固定训练集的参数，用来做测试
    net.eval()#只有前向
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)  # 前向传播
        loss = loss_fn(out, y)  # 计算损失
        if i % 50 == 0:
            print("loss", loss.item())


