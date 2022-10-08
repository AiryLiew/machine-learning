import cv2
import numpy
import matplotlib.pyplot as plt
#根据轮廓检测判断轮廓性质

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\23.jpg")
# img=cv2.imread("img_1.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#查找轮廓
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

#边界矩形拟合（最大拟合）
x,y,w,h=cv2.boundingRect(contours[0])
# img_bound=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
# cv2.imshow("",img_bound)

#最小矩形拟合
# min_rect 的输出分别是中心点、长和宽、以及旋转角度
min_rect=cv2.minAreaRect(contours[0])
print(min_rect)
#输出四个顶点的坐标
box=cv2.boxPoints(min_rect)
print(box)
# img_Rect=cv2.drawContours(img,[numpy.int32(box)],-1,(0,0,255),2)
# cv2.imshow("",img_Rect)

#最小外接圆
center,r=cv2.minEnclosingCircle(contours[0])
x,y=center
# img_Circle=cv2.circle(img,(int(x),int(y)),int(r),(0,0,255),2)
# cv2.imshow("",img_Circle)

#椭圆拟合
Ellipse=cv2.fitEllipse(contours[0])
print(Ellipse)
img_ellipse=cv2.ellipse(img,Ellipse,color=(0,0,255),thickness=2)
# cv2.imshow("",img_ellipse)
# cv2.waitKey(0)

#1.边界矩形的宽高比
ratio=w/h
print("边界矩形宽高比",ratio)

#2.轮廓面积与边界矩形的面积比
area=cv2.contourArea(contours[0])
rect_area=w*h
print("轮廓面积与边界矩形的面积比",area/rect_area)

# 3.轮廓面积与凸包面积的比
area=cv2.contourArea(contours[0])
hull=cv2.convexHull(contours[0])
hull_area=cv2.contourArea(hull)
print("轮廓面积与凸包面积的比",area/hull_area)

# 4.与轮廓面积相等的圆的直径
area=cv2.contourArea(contours[0])
d=numpy.sqrt(4*(area/numpy.pi))
print("与轮廓面积相等的圆的直径",d)

