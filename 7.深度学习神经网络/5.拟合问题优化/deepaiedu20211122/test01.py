import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
#收集的标量、向量、张量信息保存再指定的文件夹中
writer=SummaryWriter("./logs")
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,(3,3),(2,2),1)
        self.conv2=nn.Conv2d(6,16,(3,3),(2,2),1)
        self.fc=nn.Linear(16*7*7,10)#10

    def forward(self,x):
        conv1=self.conv1(x)
        y1=torch.relu(conv1)
        y1=torch.dropout(y1,0.5,False)
        conv2=self.conv2(y1)
        y2=torch.relu(conv2)
        y=torch.reshape(y2,[y2.size(0),-1])
        y=self.fc(y)
        #每个epoch保存一次对应权重、输出
        writer.add_histogram("weight1",self.conv1.weight,epoch)
        writer.add_histogram("conv2",conv2,epoch)
        writer.add_histogram("y2",y2,epoch)

        return y

if __name__ == '__main__':
    batch_size=100
    params="./params.pth"
    train_data=datasets.MNIST("./data",True,transforms.ToTensor(),download=True)
    test_data=datasets.MNIST("./data",False,transforms.ToTensor(),download=False)
    train_loader=DataLoader(train_data,100,shuffle=True)
    test_loader=DataLoader(test_data,100,shuffle=True)

    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net=Net().to(device)

    loss_fn=nn.CrossEntropyLoss()
    nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(net.parameters(),weight_decay=0.001)#l2正则化系数

    net.train()

    for epoch in range(100):

        net.train()
        train_acc=0
        train_loss=0
        for i,(x,y) in enumerate(train_loader):
            x=x.to(device)
            y=y.to(device)
            out=net(x)

            loss=loss_fn(out,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * batch_size
            out = torch.argmax(out, 1)
            train_acc += (out == y).sum().item()

        train_avgloss = train_loss / len(train_data)
        train_avgacc = train_acc / len(train_data)


        net.eval()
        eval_loss=0
        eval_Acc=0
        for i,(x,y) in enumerate(test_loader):
            x=x.to(device)
            y=y.to(device)
            out=net(x)
            loss=loss_fn(out,y)

            eval_loss+=loss.item()*batch_size
            out=torch.argmax(out,1)
            eval_Acc+=(out==y).sum().item()

        test_avgloss=eval_loss/len(test_data)
        test_avgacc=eval_Acc/len(test_data)
        print(test_avgloss,test_avgacc)

        writer.add_scalars("loss",{"train_avgloss":train_avgloss,"test_avgloss":test_avgloss})
        writer.add_scalars("acc",{"train_avgacc":train_avgacc,"test_avgacc":test_avgacc})
