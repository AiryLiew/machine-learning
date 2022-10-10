import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,explained_variance_score

class Net(nn.Module):
    def __init__(self,in_c1,out_c1,out_c2,out_c3,out_c4):
        super(Net, self).__init__()
        self.conv_1=nn.Sequential(
            nn.Conv1d(in_c1,out_c1,3,1,0),
            nn.ReLU()#7
        )

        self.conv_2=nn.Sequential(
            nn.Conv1d(out_c1,out_c2,3,1,0),
            nn.ReLU()#5
        )

        self.conv_3=nn.Sequential(
            nn.Conv1d(out_c2,out_c3,3,1,0),
            nn.ReLU()#3
        )

        self.conv_4=nn.Sequential(
            nn.Conv1d(out_c3,out_c4,3,1,0),
            nn.ReLU()#1
        )

    def forward(self, x):
        y=self.conv_1(x)
        y=self.conv_2(y)
        y=self.conv_3(y)
        y=self.conv_4(y)
        return y

if __name__ == '__main__':
    net=Net(1,256,512,256,1)
    train_data=torch.load("./train.data")
    test_data=torch.load("./test.data")
    loss_fn=nn.MSELoss()
    optim=torch.optim.Adam(net.parameters())

    net.train()
    for epoch in range(5):
        #i:0~591
        for i in range(len(train_data)-9):
            x=train_data[i:i+9]
            y=train_data[i+9:i+10]
            x=x.reshape(-1,1,9)
            y=y.reshape(-1,1,1)
            out=net(x)
            loss=loss_fn(out,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print("epoch:{},loss:{:.3f}".format(epoch,loss.item()))
        torch.save(net.state_dict(),"./waveform_params.pth")

    net.eval()
    label=[]
    output=[]
    count=[]
    plt.ion()
    for i in range(len(test_data)-9):
        x = train_data[i:i + 9]
        y = train_data[i + 9:i + 10]
        x = x.reshape(-1, 1, 9)
        y = y.reshape(-1, 1, 1)
        out = net(x)
        loss = loss_fn(out, y)
        label.append(y.numpy().reshape(-1))
        output.append(out.data.numpy().reshape(-1))
        count.append(i)
        plt.clf()
        label_icon,=plt.plot(count,label,linewidth=1,color="blue")
        output_icon,=plt.plot(count,output,linewidth=1,color="red")
        plt.legend([label_icon,output_icon],["label","output"],loc="upper right",fontsize=10)
        plt.pause(0.001)

    plt.savefig("./img.pdf")
    plt.ioff()
    plt.show()

    r2=r2_score(label,output)
    variance=explained_variance_score(label,output)
    print(r2,variance)