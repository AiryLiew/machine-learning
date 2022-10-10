import cv2
#形状匹配、轮廓比较
img1=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\22.jpg")
img2=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\21.jpg")

gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret1,thresh1=cv2.threshold(gray1,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cont1,_=cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret2,thresh2=cv2.threshold(gray2,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cont2,_=cv2.findContours(thresh2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#求的是差距
ret=cv2.matchShapes(cont1[0],cont2[0],cv2.CONTOURS_MATCH_I2,parameter=0)
print(ret)
