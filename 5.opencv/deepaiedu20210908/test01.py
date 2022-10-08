import cv2
import numpy
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\1.jpg")
#卷积滤波操作
kernel=numpy.float32([[-1,1,1],[0,0,-1],[-1,0,1]])#3*3的卷积核
dst=cv2.filter2D(img,-1,kernel)#-1代表三个通道都进行操作
# cv2.imshow("",dst)
# cv2.waitKey(0)

#低通滤波/平滑滤波：用于去噪
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\132.jpg")
img2=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\133.jpg")
#1.均值滤波
mean=cv2.blur(img,ksize=(5,5))
#2.中值滤波
median=cv2.medianBlur(img,ksize=5)#核的宽高一样的时候，可以使用一个值代替
#3.高斯滤波。sigmaX是X轴上的标准差
Gaussian=cv2.GaussianBlur(img,ksize=(5,5),sigmaX=3,sigmaY=3)
#4.双边滤波:和高斯滤波相比，双边滤波不但考虑了像素空间上的差异，还考虑了颜色空间上的差异，具有保边性
bilateral=cv2.bilateralFilter(img2,33,77,77)
#
# cv2.imshow("",img)
# cv2.imshow("",mean)
# cv2.imshow("",median)
# cv2.imshow("",Gaussian)
cv2.imshow("",bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

