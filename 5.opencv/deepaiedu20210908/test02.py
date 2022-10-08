import cv2
import numpy
import matplotlib.pyplot as plt
#高通滤波
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\1.jpg")
#锐化操作：增强图形轮廓、边缘
#1、自定义锐化核
kernel=numpy.float32([[0,-1,0],[-1,5,-1],[0,-1,0]])
dst1=cv2.filter2D(img,-1,kernel)
# cv2.imshow("",img)
# cv2.imshow("",dst1)
# cv2.waitKey(0)
#2.USM锐化
Gaussian=cv2.GaussianBlur(img,ksize=(99,99),sigmaX=3)
dst2=cv2.addWeighted(img,1,Gaussian,-1,0)#alpha*src1+beta*src2+gamma
# cv2.imshow("",dst2)
# cv2.waitKey(0)

#高通滤波：提取图形轮廓

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\6.jpg",0)
#1.索贝尔算子,#5*5的核与二阶导组合相当于沙尔算子，1*1的核与二阶导的组合相当于拉普拉斯算子
sobelX=cv2.Sobel(img,-1,dx=2,dy=0,ksize=5)
sobelY=cv2.Sobel(img,-1,dx=0,dy=2,ksize=5)
#增强梯度
sobelXABS=cv2.convertScaleAbs(sobelX,1,1)#src*alpha+beta，计算绝对值，将结果转成8位
sobelYABS=cv2.convertScaleAbs(sobelY,1,1)#src*alpha+beta，计算绝对值，将结果转成8位

soble=cv2.addWeighted(sobelXABS,0.5,sobelYABS,0.5,0)

#2.沙尔算子
scharrX=cv2.Scharr(img,-1,dx=1,dy=0)
scharrY=cv2.Scharr(img,-1,dx=0,dy=1)
scharrXABS=cv2.convertScaleAbs(scharrX)
scharrYABS=cv2.convertScaleAbs(scharrY)
scharr=cv2.addWeighted(scharrXABS,0.5,scharrYABS,0.5,0)

#3.拉普拉斯算子,是索贝尔算子求二阶段的结果
Laplacian=cv2.Laplacian(img,-1)

title=["IMG","sobelXABS","sobelYABS","soble","scharrXABS","scharrYABS","scharr","Laplacian"]
images=[img,sobelXABS,sobelYABS,soble,scharrXABS,scharrYABS,scharr,Laplacian]

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i],cmap="gray")
    plt.title(title[i])
    plt.xticks(),plt.yticks()
plt.show()