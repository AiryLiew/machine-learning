from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class Sample_Data(Dataset):
    def __init__(self,img_path,label_path):
        self.img_path = img_path
        self.dataset=open(label_path).readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.img_path, strs[0])
        label = torch.tensor([(float(strs[1])-0.5)/0.5,(float(strs[2])/224-0.5)/0.5, (float(strs[3])/224-0.5)/0.5, (float(strs[4])/224-0.5)/0.5, (float(strs[5])/224-0.5)/0.5],dtype=torch.float32)
        img_data = self.data_norm(Image.open(img_path))
        return img_data, label

    def data_norm(self, x):
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )(x)

if __name__ == '__main__':

    train_img_path = r".\Dataset\Train_image"
    train_label_path = r".\Dataset\Train_label.txt"
    validate_img_path = r".\Dataset\Validate_image"
    validate_label_path = r".\Dataset\Validate_label.txt"
    save_train_path = r".\Dataset\train.data"
    save_validate_path = r".\Dataset\validate.data"
    train_data = Sample_Data(train_img_path,train_label_path)
    validate_data = Sample_Data(validate_img_path,validate_label_path)

    torch.save(train_data,save_train_path)
    torch.save(validate_data,save_validate_path)
    train_data = torch.load(save_train_path)
    dataloader = DataLoader(train_data,5,shuffle=True,num_workers=4,drop_last=True)
    for i,(img,label) in enumerate(dataloader):
        print(img.shape)
        print(label.shape)


