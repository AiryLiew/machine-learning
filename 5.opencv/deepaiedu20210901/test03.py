import cv2
#图像二值化操作
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\3.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度图
#小于等于阈值的像素置零，大于阈值的像素置为255
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#自动求取阈值：最大类间方差法
ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)#同上
print(ret)
cv2.imshow("",binary)
cv2.waitKey(0)