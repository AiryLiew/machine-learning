import torch
from torch import nn

class ResNet_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblock=nn.Sequential(
            nn.Conv2d(64,64,(3,3),(1,1),1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3),(1,1),1)
        )

    def forward(self,x):
        y=torch.relu(self.resblock(x)+x)
        return y

if __name__ == '__main__':
    x=torch.randn(1,64,112,112)
    net_block=ResNet_Block()
    y=net_block(x)
    print(y.shape)