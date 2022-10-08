import cv2
import numpy
import matplotlib.pyplot as plt
#根据轮廓查找计算轮廓内的面积、周长、重心

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\22.jpg")
# img=cv2.imread("img_1.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#查找轮廓
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

#重心
M=cv2.moments(contours[0])#传入的是最外层的轮廓矩阵
print(M)
cx=int(M["m10"]/M["m00"])
cy=int(M["m01"]/M["m00"])
print("重心",cx,cy)

#面积
Area=cv2.contourArea(contours[0])
Area2=M["m00"]
print("面积",Area)
print("面积",Area2)

#周长
arcLength=cv2.arcLength(contours[0],True)
print("周长",arcLength)

plt.imshow(img)
plt.show()