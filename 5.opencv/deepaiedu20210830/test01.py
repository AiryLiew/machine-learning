import cv2
import numpy as np
#图像显示
img=cv2.imread("images/1.jpg")
# cv2.imshow("",img)
# cv2.waitKey(0)#延迟，毫秒
print(type(img))#numpy
#黑白图
img1=np.zeros([400,400,3])
img2=np.ones([400,400,3])
#黑白渐变色
img3=np.arange(400*400*3).reshape([400,400,3])/(400*400*3)#归一化
# cv2.imshow("",img3)
# cv2.waitKey(0)#延迟，毫秒
print(np.max(img3),np.min(img3))
#各种分布图
img=np.random.rand(400,400,3)
img=np.random.randn(400,400,3)
img=np.random.normal(5,2,(400,400,3))
# cv2.imshow("",img)
# cv2.waitKey(0)#延迟，毫秒
#生成红绿蓝图片
img=np.zeros([400,400,3])
print(img,img.shape)
img[:,:,2]=255#BGR
print(img)
cv2.imshow("",img)
cv2.waitKey(0)#延迟，毫秒
cv2.imwrite("test01.jpg",img)