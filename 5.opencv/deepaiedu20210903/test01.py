import cv2
import numpy as np
from  PIL import  Image
#图像掩码操作
img1=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\1.jpg")
img2=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\7.jpg",0)
img3=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\8.jpg")
img3=Image.fromarray(img3)
img3=np.array(img3.resize((img1.shape[1],img1.shape[0])))

ret,mask=cv2.threshold(img2,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# cv2.imshow("",mask)
# cv2.waitKey(0)
#图像的白色部分是掩码/蒙板
mask_inv=cv2.bitwise_not(mask)
# cv2.imshow("",mask_inv)
# cv2.waitKey(0)
#做蒙板操作，白色部分保留，黑色部分去除
img_and=cv2.bitwise_and(img1,img1,mask=mask_inv)
# cv2.imshow("",img_and)
# cv2.waitKey(0)

#求并集，加法
new_img=np.uint8(0.8*img1+0.2*img3-10)
# cv2.imshow("",new_img)
# cv2.waitKey(0)
new_img2=cv2.addWeighted(img1,0.7,img3,0.3,0)
cv2.imshow("",new_img)
cv2.waitKey(0)