import torch
import numpy
a=torch.tensor(1,dtype=torch.float32,device="cuda",requires_grad=True)
print(a)

#提取tensor中的标量数据
print(a.item())

b=torch.rand([2,3],requires_grad=True)
print(b)

#tensor去除梯度
print(b.data)
print(b.detach())

#tensor转ndarry
print(b.data.numpy())
print(numpy.array(b.data))

#ndarry转tensor
print(torch.tensor(numpy.array(b.data)))

print(b.item())