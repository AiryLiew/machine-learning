import cv2
path="images/1.mp4"
savepath="images/2.mp4"
cap=cv2.VideoCapture(path)#创建视频对象
fourcc=cv2.VideoWriter_fourcc(*"DIVX")#创建保存图像对象
fps=cap.get(cv2.CAP_PROP_FPS)#帧率
w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)#宽
h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#高
W=cap.get(3)#宽
H=cap.get(4)#高
print(w,h,W,H)
print(fps)
out=cv2.VideoWriter(savepath,fourcc,fps,(int(w),int(h)))
while True:
    ret,frame=cap.read()#捕获每一帧的图像
    if ret:

        cv2.putText(frame,"beautiful girl 啊",(100,100),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,255),5,cv2.LINE_AA)
        cv2.imshow("", frame)
        out.write(frame)
        # cv2.waitKey(int(1000//fps*4))#每一帧的毫秒数,慢4倍播放
        if cv2.waitKey(int(1000//fps))&0xFF==ord("q"):#按“q"键结束播放
            break
    else:
        break
cap.release()#关闭相机、视频
cv2.destroyAllWindows()#关闭所有窗口

