import torch
a=torch.randn(10,12,224,224)
#把通道上的值展开到像素上，上采样过程
b=torch.pixel_shuffle(a,2)
print(b.shape)
#把像素上的值展开到通道上，下采样过程
c=torch.pixel_unshuffle(b,2)
print(c.shape)

d=torch.nn.PixelShuffle(2)
e=d(a)
print(e.shape)
f=torch.nn.PixelUnshuffle(2)
g=f(b)
print(g.shape)