import cv2
import numpy

#模板匹配

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\16.jpg")
template=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\17.jpg")
h,w,c=template.shape

# res=cv2.matchTemplate(img,template,cv2.TM_CCORR)#相关系数匹配方法，最大值匹配''
# res=cv2.matchTemplate(img,template,cv2.TM_SQDIFF)#平方差匹配方法，最小值匹配''
# res=cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)#归一化最大匹配方法，最大值匹配(0,1)''
res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF)#最大匹配方法，最大值匹配
# res=cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)#归一化相关系数匹配方法，最大值匹配(0,1)
# res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)#归一化最大匹配方法，最小值匹配(-1,1)


min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
print(min_val)
print(max_val)
print(min_loc)
print(max_loc)
#最大值匹配
cv2.rectangle(img,(max_loc[0],max_loc[1]),(max_loc[0]+w,max_loc[1]+h),color=(0,0,255),thickness=2)
#最小值匹配
# cv2.rectangle(img,(min_loc[0],min_loc[1]),(min_loc[0]+w,min_loc[1]+h),color=(0,0,255),thickness=2)

cv2.imshow("",img)
cv2.waitKey(0)
