import cv2
import matplotlib.pyplot as plt
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\8.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#绘制图像直方图
gray_Hist=cv2.calcHist([gray],channels=[0],mask=None,histSize=[256],ranges=[0,255])
b_hist=cv2.calcHist([img],[0],None,[256],[0,255])
g_hist=cv2.calcHist([img],[1],None,[256],[0,255])
r_hist=cv2.calcHist([img],[2],None,[256],[0,255])

plt.plot(gray_Hist,"gray",label="gray")
plt.plot(b_hist,"b",label="B")
plt.plot(g_hist,"g",label="G")
plt.plot(r_hist,"r",label="R")
plt.show()
