import cv2
import numpy as np
#画多边形
img=np.zeros([400,400,3])
pt=np.array([[100,100],[200,150],[300,150],[300,300]])
cv2.polylines(img,[pt],True,(0,0,255),3)
cv2.imshow("",img)
cv2.waitKey(0)