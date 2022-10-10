import cv2
import numpy
import matplotlib.pyplot as plt
#根据轮廓查找进行轮廓近似比较

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\22.jpg")
# img=cv2.imread("img_1.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#查找轮廓
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE,offset=(0,0))


#轮廓近似
# epsilon表示原始的轮廓曲线与近似曲线之间的最大距离
approx=cv2.approxPolyDP(contours[0],epsilon=20,closed=True)
print(approx,approx.shape)

cv2.drawContours(img,contours[0],-1,(0,0,255),3)
cv2.drawContours(img,[approx],-1,(0,255,0),3)

cv2.imshow("",img)
cv2.waitKey(0)

