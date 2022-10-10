import cv2

#直方图反向投影
#寻找感兴趣区域：ROI
#感兴趣区域
roi=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\30.jpg")
hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#目标图像
target=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\16.jpg")
hsv_target=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

#1.统计hsv_roi的直方图
hist_roi=cv2.calcHist([hsv_roi],[0,1],None,[180,256],[0,179,0,255])

#2.直方图归一化
hist_roi=cv2.normalize(hist_roi,0,255,cv2.NORM_MINMAX)

#3.反向投影
BackProject=cv2.calcBackProject([hsv_target],[0,1],hist_roi,[0,179,0,255],1)

#4.卷积操作
kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(5,5))
BackProject=cv2.filter2D(BackProject,-1,kernel)

#5.图像二值化操作
ret,thresh=cv2.threshold(BackProject,50,255,cv2.THRESH_BINARY)

#6.抠除目标图中感兴趣区域
threshold=cv2.merge((thresh,thresh,thresh))
res=cv2.bitwise_and(target,threshold)
# cv2.imshow("",roi)
# cv2.imshow("",target)
# cv2.imshow("",BackProject)
# cv2.imshow("",thresh)
cv2.imshow("",res)
cv2.waitKey(0)
