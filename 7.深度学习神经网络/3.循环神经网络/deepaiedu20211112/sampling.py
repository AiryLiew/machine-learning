import os
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class Sampling(Dataset):
    def __init__(self,image_root):
        super().__init__()
        self.images=[]
        self.labels=[]
        for filename in os.listdir(image_root):
            x=os.path.join(image_root,filename)
            y=filename.split(".")[0]
            self.images.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path=self.images[index]
        image_data=Image.open(image_path)
        image_norm=self.transform(image_data)
        label_data=self.labels[index]
        label_onehot=self.one_hot(label_data)
        return image_norm,label_onehot

    def transform(self,x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,],[0.5,])
        ])(x)

    def one_hot(self,data):
        array=list(map(int,list(data)))
        zeros_arr=torch.zeros(4,10)
        zeros_arr[torch.arange(4),array]=1
        return zeros_arr

if __name__ == '__main__':
    samping=Sampling("./image_data")
    dataloader=DataLoader(samping,8,shuffle=True)
    for i,(img,label) in enumerate(dataloader):
        print(img.shape)
        print(label.shape)
        print(i)