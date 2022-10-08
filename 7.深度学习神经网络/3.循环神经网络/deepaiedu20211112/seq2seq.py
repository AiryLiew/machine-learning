import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sampling import Sampling

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.gru=nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

    def forward(self,x):
        #[N,3,60,240]-->[N,180,240]-->[N,240,180],NCHW-->NVS-->NSV
        x=x.reshape(x.size(0),3*60,240).permute(0,2,1)
        #[N,240,180]-->[N*240,180],nsv-->nv
        x=x.reshape(-1,180)
        #[N*240,180]*[180,128]=[N*240,128]
        fc=self.fc(x)
        ##[N*240,128]-->[N,240,128]
        nsv=fc.reshape(-1,240,128)
        gru,h_t=self.gru(nsv,None)
        #[N,240,128]--[N,128]
        en_out=gru[:,-1,:]
        return en_out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.gru=nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc=nn.Linear(128,10)

    def forward(self,x):
        #[N,128]-->[N,1,128]
        # x=x.reshape(x.size(0),1,-1)
        x=x.unsqueeze(1)
        #[N,1,128]-->[N,4,128]
        x=x.expand(-1,4,128)
        gru,h_t=self.gru(x,None)
        #[N,4,128]-->[N*4,128]
        gru_y=gru.reshape(-1,128)
        #[N*4,128]*[128,10]=[N*4,10]
        fc_y=self.fc(gru_y)
        #[N*4,10]-->[N,4,10]
        de_out=fc_y.reshape(-1,4,10)
        return de_out


class Main_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
    def forward(self,x):
        en_out=self.encoder(x)
        de_out=self.decoder(en_out)
        return de_out

if __name__ == '__main__':
    BATCH=64
    EPOCH=100
    save_params=r"params/seq2seq.pth"
    if not os.path.exists("./params"):
        os.mkdir("./params")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=Main_net().to(device)

    loss_fn=nn.MSELoss()
    optim=torch.optim.Adam(net.parameters())

    if os.path.exists((save_params)):
        net.load_state_dict(torch.load(save_params))
    else:
        print("random params")

    train_data=Sampling("./image_data")
    train_loader=DataLoader(train_data,BATCH,shuffle=True,num_workers=2)

    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            x=x.to(device)
            y=y.to(device)
            output=net(x)
            # print(output.shape)
            loss=loss_fn(output,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i%5==0:
                label=torch.argmax(y.cpu(),2).detach().numpy()
                out=torch.argmax(output.cpu(),2).detach().numpy()

                ACC=np.sum(out==label)/(BATCH*4)
                print("epoch:{},i{},loss:{:.3f},acc:{}%"
                      .format(epoch,i,loss.item(),ACC*100))
                print("label:",label[0])
                print("output:",out[0])
        torch.save(net.state_dict(),save_params)