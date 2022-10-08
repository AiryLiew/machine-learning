import cv2
import numpy as np
#插值算法
img1=cv2.imread("5、opencv\deepaiedu20210830\images\6.jpg")
img2=cv2.imread("5、opencv\deepaiedu20210830\images\1.jpg")
img3=cv2.imread("5、opencv\deepaiedu20210830\images\8.jpg")
cv2.imshow("",img1)
cv2.waitKey(0)
rows,clos,channels = img1.shape
img_resize1=cv2.resize(img1,(clos*4,rows*4))
img_resize2=cv2.resize(img1,(clos*4,rows*4),cv2.INTER_NEAREST)#邻近插值
img_resize3=cv2.resize(img1,(clos*4,rows*4),cv2.INTER_LINEAR)#双线性插值
img_resize4=cv2.resize(img1,(clos*4,rows*4),cv2.INTER_AREA)#局部像素重采样
img_resize5=cv2.resize(img1,(clos*4,rows*4),cv2.INTER_CUBIC)#4*4领域内3次插值
img_resize6=cv2.resize(img1,(clos*4,rows*4),cv2.INTER_LANCZOS4)#8*8领域内兰索斯插值

#效率：邻近插值》双线性》双三次》兰索斯，效率和效果是相反的
# cv2.imshow("",img_resize1)
# cv2.waitKey(0)
# cv2.imshow("",img_resize6)
# cv2.waitKey(0)

img3=cv2.resize(img3,(img2.shape[1],img2.shape[0]))
img_add=cv2.add(np.uint8(img2*0.8),np.uint8(img3*0.2))
img_sub=cv2.subtract(np.uint8(img2*0.8),np.uint8(img3*0.2))
img_mul=cv2.multiply(np.uint8(img2),np.uint8(img3*0.01))
img_div=cv2.divide(np.uint8(img2),np.uint8(img3*0.01))
cv2.imshow("",img_mul)
cv2.waitKey(0)