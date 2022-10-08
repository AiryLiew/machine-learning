import cv2
import numpy

#轮廓查找与绘制
#基于canny算子
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\22.jpg")
# img=cv2.imread("img_1.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#查找轮廓
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

print(contours)
for i in range(numpy.shape(contours)[0]):
    print(len(contours[i]))#每个轮廓向量长度的数量
print(numpy.shape(contours[0]))#表示是第一个轮廓矩阵的形状,也是最外层的轮廓

print(hierarchy)#轮廓的层级，等高线数量
print(hierarchy.shape)
#轮廓检索方式
#cv2.RETR_EXTERNAL 只检测外轮廓
#cv2.RETR_TREE 建立一个等级树结构的轮廓，包含关系
#cv2.RETR_LIST 检测多个轮廓，但是不建立等级关系
#cv2.RETR_CCOMP 建立两层等级关系的轮廓，上面的是外边界，里面的是内边界

#轮廓近似方法
# cv2.CHAIN_APPROX_NONE  存储所有的边界点
# cv2.CHAIN_APPROX_SIMPLE 压缩垂直、水平、对角线方向，只保留端点

#绘制轮廓
img_contours=cv2.drawContours(img,contours,-1,(0,255,0),2)
cv2.imshow("",img_contours)
cv2.waitKey(0)