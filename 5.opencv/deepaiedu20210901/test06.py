import cv2
import  numpy as np
import PIL.Image as Image
#图像的运算
img1=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\1.jpg")
img2=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\7.jpg")
img3=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\8.jpg")
img3=Image.fromarray(img3)
img3=np.array(img3.resize((img1.shape[1],img1.shape[0])))
new_img=np.uint8(0.8*img1+0.2*img3)
cv2.imshow("",new_img)
cv2.waitKey(0)