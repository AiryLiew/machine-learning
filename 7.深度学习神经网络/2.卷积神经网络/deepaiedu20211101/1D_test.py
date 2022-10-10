import torch
import torch.nn as nn

a= torch.randn(100,1,20)#shape:N,C,L(w/h)
conv_1d=nn.Conv1d(1,16,3,2,1)
b=conv_1d(a)
print(b.shape)
