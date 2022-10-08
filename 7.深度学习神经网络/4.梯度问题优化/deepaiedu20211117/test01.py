import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,4,(3,3),(2,2),1),
            #BN是在输出通道上做norm，需要输入输出通道
            # torch.nn.BatchNorm2d(4),
            #LN是在批次上做norm，所以需要输入图像形状
            # torch.nn.LayerNorm([1,4,2,2]),
            # IN是在批次和通道上做norm，需要输入输出通道
            # torch.nn.InstanceNorm2d(4),
            #GN是在每个组内的通道上做norm，，所以需要输入分组数和通道数
            torch.nn.GroupNorm(2,4),
            torch.nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

if __name__ == '__main__':
    x=torch.randn(1,1,4,4)
    net=Net()
    y=net(x)
    print(y.shape)