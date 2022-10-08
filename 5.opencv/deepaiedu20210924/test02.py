import cv2
import numpy
import matplotlib.pyplot as plt
#对象掩码操作

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\23.jpg")
# img=cv2.imread("img_1.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#查找轮廓
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

#创建一个0矩阵的掩码
mask=numpy.zeros(img.shape,numpy.uint8)

#使用bgr中的b通道做掩码
cv2.drawContours(mask,contours,-1,(255,0,0),-1)

pix=numpy.transpose(numpy.nonzero(mask))
print(pix)

cv2.imshow("",mask)
cv2.waitKey(0)