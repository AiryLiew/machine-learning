from torchvision import models
import torch

net=models.mobilenet_v2(True)
net1=models.vgg16(True)
net2=models.densenet121(True)
net3=models.mobilenet_v3_small(True)

#增加层数
# net.classifier=torch.nn.Sequential(
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(in_features=1280, out_features=1000, bias=True),
#     torch.nn.Linear(1000,10))
# print(net)

# 修改层数
# net.classifier[1]=torch.nn.Linear(in_features=1280, out_features=10, bias=True)
# net.classifier=torch.nn.Sequential(
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(in_features=1280, out_features=10, bias=True))
# print(net)

data=torch.randn([10,3,28,28])
# print(net(data).shape)


# net1.classifier.out=torch.nn.Linear(1000,10)
# print(net1)

# net1.classifier[6]=torch.nn.Linear(4096,10)
# print(net1)
# net2.classifier=torch.nn.Linear(in_features=1024, out_features=10, bias=True)
# print(net2)
net3.classifier[3]=torch.nn.Linear(in_features=1024, out_features=10, bias=True)
print(net3)