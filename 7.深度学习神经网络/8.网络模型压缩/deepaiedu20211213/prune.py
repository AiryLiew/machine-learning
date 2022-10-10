import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'局部剪枝'
if __name__ == '__main__':

    model = LeNet().to(device=device)
    module = model.conv1
    # named_parameters()和parameters()，前者给出网络层的名字和参数的迭代器，而后者仅仅是参数的迭代器。
    # print(list(module.named_parameters()))
    # print(list(module.parameters()))
    #生成权重剪枝的掩码，没有任何剪枝的为空。
    # print(list(module.named_buffers()))
    # print(module.weight)
    #随机非结构化剪枝：剪元素
    # prune.random_unstructured(module, name="weight", amount=0.3)
    # print(module.weight)
    # 随机结构化剪枝：剪通道
    # prune.random_structured(module, name="weight", amount=0.3,dim=0)
    # print(list(module.named_buffers()))
    # # 执行完剪枝操作后，执行prune.remove()方法会把剪枝的参数从缓冲区中删除，形成永久性剪枝。
    # prune.remove(module, 'weight')
    # print(list(module.named_parameters()))
    # # 生成权重剪枝的掩码，布尔值，1表示没有被剪枝，0表示被剪枝。
    # print(module.weight)
    #前向拦截器，由于一部分权重被置零了，再次前向计算的时候，需要拦截这部分，不需要关注。
    # print(module._forward_pre_hooks)
    # L1非结构化剪枝
    # prune.l1_unstructured(module, name="weight", amount=0.3)
    # prune.l1_unstructured(module, name="bias", amount=3)
    # print(list(module.named_parameters()))
    # print(module.weight)
    # print(module.bias)
    # print(module._forward_pre_hooks)

    #迭代结构化剪枝，剪枝一般不会一次性剪，而是迭代的剪枝:一边剪枝一边训练，迭代剪枝。
    # amount是要剪枝的数量，n是范数，dim(0,1)是输出/输入特征，输出/输入通道
    # prune.ln_structured(module, name="weight", amount=0.5, n=1, dim=0)
    # print(list(module.named_parameters()),"\n")
    # print(list(module.named_buffers()),"\n")
    # print(module.weight)


    #序列化剪枝后的模型，所有相关的张量，包括掩码缓冲区和用于计算修剪后张量的原始参数，都存储在模型的state_dict中
    # print(model.state_dict().keys())
    # 执行完剪枝操作后，执行prune.remove()方法会把剪枝的参数从缓冲区中删除，形成永久性剪枝。
    # prune.remove(module, 'weight')
    # print(list(module.named_parameters()))
    # print(list(module.named_buffers()))
    # print(module.weight)
    # print(model.state_dict().keys())

    # 减除模型中的多个参数，比如同时对线性层和卷积层参数剪枝
    # new_model = LeNet()
    # for name, module in new_model.named_modules():
    # #     # 在所有卷积层中剪掉20%的连接
    #     if isinstance(module, torch.nn.Conv2d):
    #         prune.l1_unstructured(module, name='weight', amount=0.2)
    # #     # 在所有线性层中剪掉40%的连接
    #     elif isinstance(module, torch.nn.Linear):
    #         prune.l1_unstructured(module, name='weight', amount=0.4)
    # #
    # print(list(new_model.named_parameters()))
    # print(list(new_model.named_buffers()))
    # print(dict(new_model.named_buffers()).keys())  # 查看缓存区掩码

'全局剪枝'
if __name__ == '__main__':
    """
    一种常见的、可能更强大的技术,是通过删除(例如)整个模型中最低20%的连接，
    而不是删除每个层中最低20%的连接，一次性删除整个模型。这可能导致每层的修剪百分数不同。
    """
#     model = LeNet()
#     parameters_to_prune = (
#         (model.conv1, 'weight'),
#         (model.conv2, 'weight'),
#         (model.fc1, 'weight'),
#         (model.fc2, 'weight'),
#         (model.fc3, 'weight'),
#     )
#
#     prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=0.2
#     )
#     #统计每层权重被剪枝的百分比
#     print(
#         "Sparsity in conv1.weight: {:.2f}%".format(
#             100. * float(torch.sum(model.conv1.weight == 0))
#             / float(model.conv1.weight.nelement())
#         )
#     )
#     print(
#         "Sparsity in conv2.weight: {:.2f}%".format(
#             100. * float(torch.sum(model.conv2.weight == 0))
#             / float(model.conv2.weight.nelement())
#         )
#     )
#     print(
#         "Sparsity in fc1.weight: {:.2f}%".format(
#             100. * float(torch.sum(model.fc1.weight == 0))
#             / float(model.fc1.weight.nelement())
#         )
#     )
#     print(
#         "Sparsity in fc2.weight: {:.2f}%".format(
#             100. * float(torch.sum(model.fc2.weight == 0))
#             / float(model.fc2.weight.nelement())
#         )
#     )
#     print(
#         "Sparsity in fc3.weight: {:.2f}%".format(
#             100. * float(torch.sum(model.fc3.weight == 0))
#             / float(model.fc3.weight.nelement())
#         )
#     )
# #     # 统计全部权重被剪枝的百分比
#     print(
#         "Global sparsity: {:.2f}%".format(
#             100. * float(
#                 torch.sum(model.conv1.weight == 0)
#                 + torch.sum(model.conv2.weight == 0)
#                 + torch.sum(model.fc1.weight == 0)
#                 + torch.sum(model.fc2.weight == 0)
#                 + torch.sum(model.fc3.weight == 0)
#             )
#             / float(
#                 model.conv1.weight.nelement()
#                 + model.conv2.weight.nelement()
#                 + model.fc1.weight.nelement()
#                 + model.fc2.weight.nelement()
#                 + model.fc3.weight.nelement()
#             )
#         )
#     )

# '继承父类剪枝方法'
# class FooBarPruningMethod(prune.BasePruningMethod):
#
#     def compute_mask(self, t, default_mask):
#         mask = default_mask.clone()
#         mask.view(-1)[::2] = 0
#         return mask
#
#     def foobar_unstructured(self,module, name):
#         FooBarPruningMethod.apply(module, name)
#         return module
# if __name__ == '__main__':
#     model = LeNet()
#     f=FooBarPruningMethod()
#     f.foobar_unstructured(model.conv1, name='weight')
#     print(model.conv1.weight)