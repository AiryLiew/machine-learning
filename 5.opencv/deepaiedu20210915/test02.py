import cv2
import matplotlib.pyplot as plt
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\28.jpg",0)
img_equalizeHist=cv2.equalizeHist(img)
#2.局部/自适应均衡化
clahe=cv2.createCLAHE(tileGridSize=(9,9))
img_clahe=clahe.apply(img)
#原来直方图
hist_img=cv2.calcHist([img],[0],None,[256],[0,255])
#全局均衡化直方图
hist_equalizeHist=cv2.calcHist([img_equalizeHist],[0],None,[256],[0,255])
#局部均衡化直方图
hist_clahe=cv2.calcHist([img_clahe],[0],None,[256],[0,255])

plt.subplot(231),plt.imshow(img,cmap="gray"),plt.title("img"),plt.xticks([]),plt.yticks([])
plt.subplot(232),plt.imshow(img_equalizeHist,cmap="gray"),plt.title("img"),plt.xticks([]),plt.yticks([])
plt.subplot(233),plt.imshow(img_clahe,cmap="gray"),plt.title("img"),plt.xticks([]),plt.yticks([])

plt.subplot(234),plt.plot(hist_img,color="r",label="hist_img"),plt.title("hist_img"),plt.xticks([]),plt.yticks([])
plt.subplot(235),plt.plot(hist_equalizeHist,color="g",label="hist_equalizeHist"),plt.title("hist_equalizeHist"),plt.xticks([]),plt.yticks([])
plt.subplot(236),plt.plot(hist_clahe,color="b",label="hist_clahe"),plt.title("hist_clahe"),plt.xticks([]),plt.yticks([])
plt.show()