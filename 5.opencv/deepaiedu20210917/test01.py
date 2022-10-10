import cv2
import numpy
#多目标模板匹配
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\19.jpg")
template=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\20.jpg")

h,w,c=template.shape
#最大值归一化匹配方法
res=cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
print(res)
print(res)

locs=numpy.where(res>=0.995)#返回满足条件的值的索引
print(*locs)#得到的值是位置的 h和w
print(*locs[::-1])#倒序输出，得到w和h,对于x和y
for pt in zip(*locs[::-1]):
    print(pt[0],pt[1])
    cv2.rectangle(img,(pt[0],pt[1]),(pt[0]+w,pt[1]+h),color=(0,0,255),thickness=2)
cv2.imshow("",img)
cv2.waitKey(0)