import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\6.jpg")
img=cv2.imread("1.jpg")
print(img.shape)
img1=Image.fromarray(img[:,:,::-1])
# plt.imshow(img1)
# plt.show()
#图像透视变换5
pts1=np.array([[-10,-30],[620,130],[0,300],[600,200]],dtype=np.float32)#原图像的四个点的坐标
pts2=np.array([[12,184],[451,140],[34,314],[443,157]],dtype=np.float32)#透视变换的坐标
M=cv2.getPerspectiveTransform(pts2,pts1)#创建变换矩阵,头上变换的坐标在前面，原图像坐标在后面
dst=cv2.warpPerspective(img,M,(600,200))#变换操作，输出指定大小的图像

cv2.imshow("",dst)
cv2.waitKey(0)

