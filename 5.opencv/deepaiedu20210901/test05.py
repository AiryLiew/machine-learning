import cv2
import numpy as np
import matplotlib.pyplot as plt
#自适应阈值：局部二值化
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\6.jpg",0)
# ret1,thresh1=cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
img=cv2.GaussianBlur(img,(5,5),0)
ret1,thresh1=cv2.threshold(img,70,255,cv2.THRESH_BINARY)
# cv2.imshow("",thresh1)
# cv2.waitKey(0)
thresh2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
thresh3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,3)
titles=["IMAGE","THRESH_BINARY","ADAPTIVE_THRESH_GAUSSIAN_C","ADAPTIVE_THRESH_MEAN_C"]
images=[img,thresh1,thresh2,thresh3]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks()
    plt.yticks()
plt.show()