import cv2
import numpy as np
import matplotlib.pyplot as plt
#简单阈值操作
# img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\3.jpg")
img=np.uint8(np.arange(400*400).reshape([400,400])/(400*400)*255)#转灰度图
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度图
print(img.shape)
# cv2.imshow("",img)
# cv2.waitKey(0)
#自动阈值二值化
ret1,thresh1=cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
#自动阈值反二值化
ret2,thresh2=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#截断阈值操作，小于等于阈值的像素保持原色，大于阈值的像素置为255
ret3,thresh3=cv2.threshold(img,0,255,cv2.THRESH_TRUNC|cv2.THRESH_OTSU)
#取零阈值操作，小于等于阈值像素置零，大于阈值像素保持原色
ret4,thresh4=cv2.threshold(img,0,255,cv2.THRESH_TOZERO|cv2.THRESH_OTSU)
#反取零阈值操作，小于等于阈值像素保持原色，大于阈值像素置零
ret5,thresh5=cv2.threshold(img,0,255,cv2.THRESH_TOZERO_INV|cv2.THRESH_OTSU)
titles=["IMG","THRESH_BINARY","THRESH_BINARY_INV","THRESH_TRUNC","THRESH_TOZERO","THRESH_TOZERO_INV"]
images=[img,thresh1,thresh2,thresh3,thresh4,thresh5]
# cv2.imshow("",thresh4)
# cv2.waitKey(0)
for i in range(len(titles)):
    plt.subplot(2,3,i+1)#要画的图是几行几列，当前的第几个元素
    plt.imshow(images[i],"gray")#显示当前图像
    plt.title(titles[i])#显示当前的标题
    plt.xticks([])
    plt.yticks([])
plt.show()