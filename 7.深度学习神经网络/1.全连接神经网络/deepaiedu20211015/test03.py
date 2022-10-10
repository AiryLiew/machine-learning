import torch

a=torch.randn([3,5])
print(a)


wb=torch.nn.Linear(3,5)
print(wb)
print(wb.weight)
print(wb.bias)
