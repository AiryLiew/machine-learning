import cv2
#图像翻转

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\1.jpg")
# dst=cv2.transpose(img)
# cv2.imshow("",dst)
# cv2.waitKey(0)

dst1=cv2.flip(img,flipCode=0)#沿着X轴上下翻转
# cv2.imshow("",dst1)
# cv2.waitKey(0)
dst2=cv2.flip(img,flipCode=1)#沿着Y轴左右翻转
# cv2.imshow("",dst2)
# cv2.waitKey(0)
dst3=cv2.flip(img,flipCode=-1)#沿着X轴和Y轴上下左右翻转，相当于旋转180度
cv2.imshow("",dst3)
cv2.waitKey(0)