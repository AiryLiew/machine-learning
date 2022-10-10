import cv2
import matplotlib.pyplot as plt
#canny算子
'''
步骤
彩色图像转换为灰度图像 
高斯滤波，滤除噪声点 
通过sobel算子计算图像梯度，根据梯度计算边缘幅值与角度 
非极大值抑制 
双阈值边缘连接处理 
二值化图像输出结果
'''

#1.原图转灰度图
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\18.jpg",0)

#2.高斯模糊处理
GaussianBlur=cv2.GaussianBlur(img,(15,15),3)

# 3.高亮处理，增强差异度/梯度
ScaleAbs=cv2.convertScaleAbs(GaussianBlur,alpha=8,beta=0)

#4.形态学闭操作填补漏洞
kerner=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
MORPH_CLOSE=cv2.morphologyEx(ScaleAbs,cv2.MORPH_CLOSE,kerner)

#5.canny算子提取边缘
# 通过sobel算子计算图像梯度，根据梯度计算边缘幅值与角度
# 非极大值抑制
# 双阈值边缘连接处理
Canny=cv2.Canny(MORPH_CLOSE,50,150)

# 6.边缘二值化处理

ret,thresh=cv2.threshold(Canny,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
plt.subplot(221),plt.imshow(img,cmap="gray")
plt.subplot(222),plt.imshow(ScaleAbs,cmap="gray")
plt.subplot(223),plt.imshow(Canny,cmap="gray")
plt.subplot(224),plt.imshow(thresh,cmap="gray")
plt.show()