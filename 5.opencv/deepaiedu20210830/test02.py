import cv2
import numpy as np
from PIL import Image
img=cv2.imread("images/1.jpg")
#numpy切片操作，图像通道分离拼接：b,g,r
b=img[:,:,:1]
g=img[:,:,1:2]
r=img[:,:,2:]
print(b)
print(b.shape)
# cv2.imshow("",g)
# cv2.waitKey(0)
img=np.concatenate((b,g,r),axis=2)
# cv2.imshow("",img)
# cv2.waitKey(0)
pilimg=Image.fromarray(img[:,:,::-1])#b,g,r -→ r,g,b
# pilimg.show()
#opencv自带的API操作通道分离合并
b,g,r=cv2.split(img)
print(b.shape)
# cv2.imshow("",b)
# cv2.waitKey(0)
img=cv2.merge([b,g,r])
# cv2.imshow("",img)
# cv2.waitKey(0)
#在图像模式不变的前提下，只显示某个通道上的值。
#b,g,r--0,1,2
img[...,1]=0
img[...,2]=0
print(img)
print(img.shape)
cv2.imshow("",img)
cv2.waitKey(0)