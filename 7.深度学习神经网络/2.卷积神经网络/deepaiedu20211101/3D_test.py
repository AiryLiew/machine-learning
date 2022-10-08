import torch
import torch.nn as nn

a= torch.randn(100,3,10,224,224)#shape:N,C,D,H,W
conv_3d=nn.Conv3d(3,16,(3,3,3),(2,2,2),1)
b=conv_3d(a)
print(b.shape)
