import cv2
#直线、圆、椭圆、矩形
img=cv2.imread("images/1.jpg")
cv2.line(img,(100,100),(500,500),(0,0,255),thickness=3)
cv2.circle(img,(300,300),100,(0,0,255),thickness=3)
cv2.ellipse(img,(400,400),(100,100),90,180,360,(0,0,255))
cv2.rectangle(img,(100,100),(500,500),(0,0,255),thickness=3)

cv2.imshow("",img)
cv2.waitKey(0)