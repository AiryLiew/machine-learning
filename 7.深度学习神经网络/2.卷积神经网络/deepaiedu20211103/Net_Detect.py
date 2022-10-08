import torch
from torchvision import transforms
from Net_Model import Net
import os
import PIL.Image as pimg
import PIL.ImageFont as Font
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import numpy as np

train_path = r".\Dataset\Train_image"
test_path = r".\Dataset\Test_image"
params_path = r".\Params\params.pth"
font_path = r".\Dataset\msyh.ttc"

net = Net()
net.load_state_dict(torch.load(params_path))
net.eval()
for file in os.listdir(test_path):
    # img = pimg.open("{0}/{1}".format(test_path, file))#打开图像
    img = pimg.open(os.path.join(test_path,file))
    img_array = (np.array(img,dtype=np.float32) / 255 - 0.5) / 0.5 #标准化数据【224,224,3】
    trans_array = np.transpose(img_array, [2, 0, 1])#将[H,W,C]转为[C,H,W]【3,224,224】
    tensor_array = torch.from_numpy(trans_array)#将numpy转为tensor【3,224,224】
    # tensor_array=transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    # )(img)
    batch_array = torch.unsqueeze(tensor_array, dim=0)#扩维度将[C,H,W]转为[N,C,H,W]【1,3,224,224】

    out1,out2=net(batch_array)
    c_out = out1.data.numpy()
    coord_out = out2.data.numpy()

    # out_x1 = coord_out[0][0] * 224
    # out_y1 = coord_out[0][1] * 224
    # out_x2 = coord_out[0][2] * 224
    # out_y2 = coord_out[0][3] * 224
    # out_confidence = c_out[0][0]

    out_x1 = (coord_out[0][0]*0.5+0.5) * 224
    out_y1 = (coord_out[0][1]*0.5+0.5) * 224
    out_x2 = (coord_out[0][2]*0.5+0.5) * 224
    out_y2 = (coord_out[0][3]*0.5+0.5) * 224
    out_confidence = c_out[0][0]*0.5+0.5

    print("output_coord:", out_x1, out_y1, out_x2, out_y2)
    print("output_confidences:", out_confidence)

    # img = pimg.fromarray(np.uint8((img_array*0.5+0.5)*255))
    imgdraw = draw.ImageDraw(img)

    if out_confidence>0.7:
        imgdraw.rectangle((out_x1, out_y1, out_x2, out_y2), outline="red",width=3)
        font = Font.truetype(font_path, size=15)
        imgdraw.text(xy=(out_x1, out_y1), text=str("{:.2f}".format(out_confidence)), fill="red", font=font)
    plt.title("test")
    plt.pause(1)
    plt.imshow(img)