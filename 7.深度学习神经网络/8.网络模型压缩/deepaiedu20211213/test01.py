import torch
x = torch.randn(size=(4, 3, 5, 6))
# x = torch.tensor([[1.], [2], [3]])
print(x.nelement())
print(x.shape)
print(x.reshape(-1,int(x.nelement()/x.shape[0])).shape)
print(x.reshape(x.shape[0],-1).shape)
from torchvision import models
models.densenet201(True)