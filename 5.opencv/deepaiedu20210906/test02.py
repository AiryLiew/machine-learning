import cv2
import numpy as np
import matplotlib.pyplot as plt

#图像形态学操作
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\10.jpg")
# cv2.imshow("",img)
# cv2.waitKey(0)

kernel1=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(5,5))#矩形核
kernel2=cv2.getStructuringElement(shape=cv2.MORPH_CROSS,ksize=(5,5))#交叉形核，十字形核
kernel3=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(5,5))#椭圆形核
print(kernel1)
print(kernel2)
print(kernel3)


dilate1=cv2.dilate(img,kernel1)#膨胀，对白色
erode1=cv2.erode(img,kernel1)#腐蚀，对白色而言
dilate=cv2.morphologyEx(img,cv2.MORPH_DILATE,kernel1)#膨胀，对白色而言
erode=cv2.morphologyEx(img,cv2.MORPH_ERODE,kernel1)#腐蚀，对白色而言
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel1)#梯度,用于提取轮廓

# cv2.imshow("",dilate)
# # cv2.imshow("",erode)
# # cv2.imshow("",gradient)
# cv2.waitKey(0)
#
img2=cv2.imread("2.jpg")
MORPH_OPEN=cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel1)#开操作，先腐蚀，后膨胀，用于去除白色噪声
MORPH_CLOSE=cv2.morphologyEx(img2,cv2.MORPH_CLOSE,kernel1)#闭操作，先膨胀，后腐蚀，用于填补黑色漏洞
MORPH_TOPHAT=cv2.morphologyEx(img2,cv2.MORPH_TOPHAT,kernel1)#顶帽/礼帽操作，原图-开操作，用于获取白色噪声
MORPH_BLACKHAT=cv2.morphologyEx(img2,cv2.MORPH_BLACKHAT,kernel1)#黑帽操作，闭操作-原图，用于获取黑色漏洞

# title=["Image","dilate1","erode1","dilate","erode","gradient"]
# images=[img,dilate1,erode1,dilate,erode,gradient]
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i],cmap="gray")
#     plt.title(title[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

title2=["Image","MORPH_OPEN","MORPH_CLOSE","MORPH_TOPHAT","MORPH_BLACKHAT"]
images2=[img2,MORPH_OPEN,MORPH_CLOSE,MORPH_TOPHAT,MORPH_BLACKHAT]
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images2[i],cmap="gray")
    plt.title(title2[i])
    plt.xticks([])
    plt.yticks([])
plt.show()