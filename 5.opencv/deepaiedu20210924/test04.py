import cv2
import numpy
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\26.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#1.图像二值化
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

#2.图像去噪
kernel=numpy.ones((3,3),dtype=numpy.uint8)
open_img=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

#3.估计背景区域,膨胀
bg=cv2.dilate(open_img,kernel,iterations=3)
# cv2.imshow("",bg)
# cv2.waitKey(0)

#4.估计前景区域
dist=cv2.distanceTransform(open_img,cv2.DIST_L2,3)


ret,fg=cv2.threshold(dist,dist.max()*0.5,255,cv2.THRESH_BINARY)
# cv2.imshow("",dist)
# cv2.waitKey(0)

# 5.估计未知区域
unknow=cv2.subtract(numpy.uint8(bg),numpy.uint8(fg))
cv2.imshow("",unknow)
cv2.waitKey(0)

# 6.连通域处理：标记类别，计算中心
#输出所有的连通域的数量、和每一个像素的类别标记
num_label,markers=cv2.connectedComponents(numpy.uint8(fg))

print(markers)

markers=markers+1
markers[unknow==255]=0


#7.分水岭
water=cv2.watershed(img,markers)
img[markers==-1]=(0,0,255)
cv2.imshow("",img)
cv2.waitKey(0)