import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #序列构造器内的程序会自上而下的运行
        self.fc1=torch.nn.Sequential(
        torch.nn.Linear(784,256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU())

        torch.nn.LeakyReLU()
        torch.nn.PReLU()


        self.fc2=torch.nn.Sequential(
        torch.nn.Linear(256,64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU())

        #输出层没有BN层
        self.fc3=torch.nn.Linear(64,10)
        # self.fc3 = torch.nn.Sequential(
        #     torch.nn.Linear(64, 10),
        #     torch.nn.Sigmoid())


    def forward(self,x):
        # N,C,H,W-->N,V
        x=torch.reshape(x,[x.size(0),-1])
        #X*W
        # print(x)
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
    #参数、网络的保存路径
    if not os.path.exists("./save_params_net"):
        os.mkdir("./save_params_net")
    save_params=r"./save_params_net/params.pth"#.pth\.pt\.pkl\.pk
    save_net=r"./save_params_net/net.pth"
    # 数据初始化方法:(0,255)-->(-1,1)TENSOR
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

    #加载数据:采样
    train_loader=DataLoader(train_data,100,True,num_workers=4)
    test_loader=DataLoader(test_data,100,True,num_workers=4)

    #指定设备
    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    #实例化网络到具体设备
    # net=Net().to(device)
    #判断保存的参数是否存在，如果存在 就加载参数
    # if os.path.exists(save_params):
    #     net.load_state_dict(torch.load(save_params))#只加载参数
    #     print("加载参数成功！")
    # else:
    #     print("NO params!")

    # #判断保存的网络是否存在，如果存在就加载网络和参数
    if os.path.exists(save_net):
        net=torch.load(save_net).to(device)#加载参数和网络
    else:
        print("NO params!")


    #定义损失函数
    loss_fn=torch.nn.CrossEntropyLoss()#多分类（类别大于二）交叉熵损失函数，自动会对标签做one-hot，对输出做softmax
    torch.nn.BCELoss()#二分类交叉熵损失函数,不会对标签自动做one-hot，需要对输出层做sigmoid激活
    torch.nn.BCEWithLogitsLoss()#二分类损失函数,不会对标签自动做one-hot，和BCELoss相比，自动会对输出层做sigmoid激活
    torch.nn.MSELoss()#均方差损失函数，需要对标签做one-hot，对输出层没有要求
    #定义优化器
    optim=torch.optim.Adam(net.parameters())
    # optim=torch.optim.SGD(net.parameters(),0.001)
    # plt.ion()
    a = []
    b = []
    train=0

    if train:
        net.train()
        for epoch in range(1):
            for i,(x,y) in enumerate(train_loader):
                x=x.to(device)
                y=y.to(device)
                out=net(x)#前向传播
                # print(out.shape)
                # print(y.shape)
                loss=loss_fn(out,y)#计算损失
                #梯度下降三部曲，反向传播
                optim.zero_grad()#情况之前的梯度
                loss.backward()#重新计算当前的梯度
                optim.step()#沿着当前的梯度更新一步

                if i%50==0:
                    print("loss",loss.item())

            if epoch % 1 == 0:
                torch.save(net.state_dict(), save_params)  # 只保存参数，不保存网络
                torch.save(net, save_net)  # 将参数和网络一起保存
        #         a.append(i)
        #         b.append(loss.item())
        #         plt.clf()
        #         plt.plot(a,b)
        #         plt.pause(0.1)
        # plt.ioff()
        # plt.show()

    #评估模式，做测试，固定训练集的参数，用来做测试
    net.eval()#只有前向
    eval_loss=0
    eval_acc=0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)  # 前向传播
        loss = loss_fn(out, y)  # 计算损失
        if i % 50 == 0:
            print("loss", loss.item())
        #计算总的损失
        eval_loss+=loss.item()*y.size(0)
        #计算总的精度:取最大值的索引、从GPU取出到CPU,节约GPU资源
        eval_acc+=(y==torch.argmax(out,1)).cpu().sum().item()
    #计算平均损失
    avg_loss=eval_loss/len(test_data)
    #计算平均精度
    avg_acc=eval_acc/len(test_data)

    print(avg_acc,avg_loss)

