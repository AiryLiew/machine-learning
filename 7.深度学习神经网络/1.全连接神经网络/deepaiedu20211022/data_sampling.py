from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from PIL import Image

class My_Dataset(Dataset):
    def __init__(self,main_dir):
        self.dataset=[]
        #读取本地文件夹中文件
        for i,cls_filename in enumerate(os.listdir(main_dir)):
            for img_data in os.listdir(os.path.join(main_dir,cls_filename)):
                x=os.path.join(main_dir,cls_filename,img_data)
                y=i
                self.dataset.append([x,y])
                # print(self.dataset)

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        #从数据集中随机获取一个数据
        img,label=self.dataset[index]
        img_data=self.data_process(Image.open(img))
        return img_data,label
    def data_process(self,x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,],std=[0.5,])
        ])(x)
if __name__ == '__main__':
    train_dataset = My_Dataset(r"E:\pycharmprojects\dataset\MNIST\train")
    test_dataset = My_Dataset(r"E:\pycharmprojects\dataset\MNIST\test")
    train_loader=DataLoader(train_dataset,100,shuffle=True)
    for img,label in train_loader:
        print(img.shape)
        print(label.shape)