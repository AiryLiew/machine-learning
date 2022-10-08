import cv2
import numpy
import matplotlib.pyplot as plt
#凸包检测

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\23.jpg")
# img=cv2.imread("img_1.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#查找轮廓，注意：一定要使用简单的轮廓近似方法：CHAIN_APPROX_SIMPLE
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

#根据轮廓线查找凸包曲线
hull=cv2.convexHull(contours[0])

#检测一个曲线是不是凸的，返回True、False
print(cv2.isContourConvex(contours[0]))#检测轮廓线是否是凸的
print(cv2.isContourConvex(hull))#检测凸包是否是凸的

#绘制凸包曲线
cv2.drawContours(img,[hull],-1,(0,0,255),2)
cv2.imshow("",img)
cv2.waitKey(0)