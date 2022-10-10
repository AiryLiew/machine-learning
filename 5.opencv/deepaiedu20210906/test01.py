import cv2
import numpy as np

#图像仿射变换
img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\6.jpg")
rows,cols,channels=img.shape
#原始图像
M0=np.float32([[1,0,0],[0,1,0]])
#平移
M1=np.float32([[1,0,20],[0,1,80]])#沿着X轴平移20，沿着Y轴平移80
dst0=cv2.warpAffine(img,M0,(cols,rows))#原图，输出的宽和高  宽：列，高：行
dst1=cv2.warpAffine(img,M1,(cols,rows))#平移，输出的宽和高,相当于裁剪图像  宽：列，高：行
# cv2.imshow("",dst0)
# cv2.imshow("",dst1)
# cv2.waitKey(0)

M2=np.float32([[0.5,0,0],[0,0.5,0]])#缩放，按着X和Y轴的缩放倍数缩放，
dst2=cv2.warpAffine(img,M2,(cols//2,rows//2))#缩放，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst2)
# cv2.waitKey(0)

#center:旋转中心，angle：正数表示逆时针角度，scale表示输出的缩放
M3=cv2.getRotationMatrix2D(center=(cols//2,rows//2),angle=45,scale=1/np.sqrt(2))
dst3=cv2.warpAffine(img,M3,(cols,rows))#旋转，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst3)
# cv2.waitKey(0)

M4=np.float32([[1,-0.5,0],[0,1,0]])#沿着X轴倾斜0.5倍，正负号决定向左向右倾斜
dst4=cv2.warpAffine(img,M4,(cols,rows))#倾斜，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst4)
# cv2.waitKey(0)

M5=np.float32([[1,0,0],[0.5,1,0]])#沿着Y轴倾斜0.5倍，正负号决定向左向右倾斜
dst5=cv2.warpAffine(img,M5,(cols,rows))#倾斜，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst5)
# cv2.waitKey(0)

#翻转过后的图像超出了原有的坐标，所以需要偏移还原坐标后才能看见
M6=np.float32([[-1,0,cols],[0,-1,rows]])#沿着X轴和Y轴同时翻转，相当于旋转180度
dst6=cv2.warpAffine(img,M6,(cols,rows))#倾斜，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst6)
# cv2.waitKey(0)


M7=np.float32([[1,0,0],[0,-1,rows]])#沿着X轴翻转，上下翻转
dst7=cv2.warpAffine(img,M7,(cols,rows))#倾斜，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst7)
# cv2.waitKey(0)

M8=np.float32([[-1,0,cols],[0,1,0]])#沿着Y轴翻转，左右翻转
dst8=cv2.warpAffine(img,M8,(cols,rows))#倾斜，输出的宽和高  宽：列，高：行
# cv2.imshow("",dst8)
# cv2.waitKey(0)