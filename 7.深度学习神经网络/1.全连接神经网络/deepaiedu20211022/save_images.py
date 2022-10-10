from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

if __name__ == '__main__':
    train_imgs=r"E:\pycharmprojects\dataset\MNIST\train"
    test_imgs=r"E:\pycharmprojects\dataset\MNIST\test"

    transf=transforms.Compose([transforms.ToTensor()])
    train_data=datasets.MNIST("./data",train=True,transform=transf,download=True)
    test_data=datasets.MNIST("./data",train=False,transform=transf,download=False)

    train_loader=DataLoader(train_data,1,True,num_workers=4)
    test_loader=DataLoader(test_data,1,True,num_workers=4)
    '''
    1、判断一个路径是否存在
    os.path.exists()
    2、创建单层目录
    os.mkdir()
    3、创建多层目录
     os.makedirs()
    '''
    for i,(x,y) in enumerate(train_loader):
        # print(x.shape)
        # print(y.shape)
        y=str(y.item())
        if not os.path.exists(os.path.join(train_imgs,y)):
            os.makedirs(os.path.join(train_imgs,y))
        #默认保存为RGB格式
        save_image(x,"{}/{}.jpg".format(os.path.join(train_imgs,y),y+"."+str(i)))
    print("训练集数据保存完成！")
    for i,(x,y) in enumerate(test_loader):
        # print(x.shape)
        # print(y.shape)
        y=str(y.item())
        if not os.path.exists(os.path.join(test_imgs,y)):
            os.makedirs(os.path.join(test_imgs,y))
        #默认保存为RGB格式
        save_image(x,"{}/{}.jpg".format(os.path.join(test_imgs,y),y+"."+str(i)))
    print("测试集数据保存完成！")