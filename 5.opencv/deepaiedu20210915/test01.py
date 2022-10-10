import cv2
import matplotlib.pyplot as plt

#直方图均衡化：类似于放大直方图的方差
#1、全局直方图均衡化
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\28.jpg",0)
#做全局均衡化
dst=cv2.equalizeHist(img)

#原直方图
hist_img=cv2.calcHist([img],[0],None,[256],[0,255])
#均衡化后的直方图
hist_dst=cv2.calcHist([dst],[0],None,[256],[0,255])

plt.subplot(221),plt.imshow(img,cmap="gray"),plt.title("img"),plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(dst,cmap="gray"),plt.title("dst"),plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.plot(hist_img,color="r",label="hist_img"),plt.title("img"),plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.plot(hist_dst,color="b",label="hist_dst"),plt.title("img"),plt.xticks([]),plt.yticks([])
plt.show()