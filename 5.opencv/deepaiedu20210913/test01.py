import cv2
import numpy
import matplotlib.pyplot as plt
#使用numpy进行傅里叶变换4

img=cv2.imread(r"E:\pycharmprojects\pythonProject\deepaiedu20210830\images\1.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows,clos=gray.shape

#1.快速傅里叶变换：FFT。空域--》频域
fft=numpy.fft.fft2(gray)#傅里叶变换


#2.中心化排列频域信号：把低频信号移动到图像中心,为了方便切片操作
fftshift=numpy.fft.fftshift(fft)
#绝对值最大的频率信号
print(numpy.max(numpy.abs(fftshift)))
#绝对值最小的频率信号
print(numpy.min(numpy.abs(fftshift)))
#获取频率震幅图，通过log函数的压缩，获取频率变化趋势变化图
logabs=numpy.log(numpy.abs(fftshift))
print(numpy.max(logabs),numpy.min(logabs))

#3.滤波操作：高通滤波（去低频，保高频）
t=100
fftshift[rows//2-t:rows//2+t,clos//2-t:clos//2+t]=0

#4.频域信号去中心化：对保留的频率信号进行位置还原
ifftshift=numpy.fft.ifftshift(fftshift)

# 5.傅里叶逆变换：频域--》空域
ifft=numpy.fft.ifft2(ifftshift)
print(ifft)

# 6.二维向量取模
img_fft=numpy.abs(ifft)
#
# plt.figure(figsize=(10,10))
# plt.subplot(221),plt.imshow(gray,cmap="gray"),plt.title("gray"),plt.xticks([]),plt.yticks([])
#
# plt.subplot(222),plt.imshow(logabs,cmap="gray"),plt.title("logabs"),plt.xticks([]),plt.yticks([])
# #
# plt.subplot(223),plt.imshow(img_fft,cmap="gray"),plt.title("img_fft"),plt.xticks([]),plt.yticks([])
# plt.subplot(224),plt.imshow(img_fft),plt.title(""),plt.xticks([]),plt.yticks([])
# plt.show()

#1.离散傅里叶变换：DFT。空域--》频域
dft=cv2.dft(numpy.float32(gray),flags=cv2.DFT_COMPLEX_OUTPUT)

#2.中心化
fftshift=numpy.fft.fftshift(dft)
#绝对值最大的频率信号
print(numpy.max(numpy.abs(fftshift)))
#绝对值最小的频率信号
print(numpy.min(numpy.abs(fftshift)))
#获取频率震幅图，通过log函数的压缩，获取频率变化趋势变化图
logabs=numpy.log(cv2.magnitude(fftshift[:,:,0],fftshift[:,:,1]))
print(numpy.max(logabs),numpy.min(logabs))
# 3.低通滤波操作（去高频，保低频）
mask=numpy.zeros((rows,clos,2),dtype=numpy.uint8)
#得到掩码图,要保留的图像部分
t=100
mask[rows//2-t:rows//2+t,clos//2-t:clos//2+t]=1
fftshift=fftshift*mask

# 4.去中心化：将保留的信号位置还原
ifftshift=numpy.fft.ifftshift(fftshift)

# 5.傅里叶逆变换：频域--空域
idft=cv2.idft(ifftshift)

#6.二维向量取模
img_dft=cv2.magnitude(idft[:,:,0],idft[:,:,1])

plt.figure(figsize=(10,10))
plt.subplot(221),plt.imshow(gray,cmap="gray"),plt.title("gray"),plt.xticks([]),plt.yticks([])

plt.subplot(222),plt.imshow(logabs,cmap="gray"),plt.title("logabs"),plt.xticks([]),plt.yticks([])
#
plt.subplot(223),plt.imshow(img_dft,cmap="gray"),plt.title("img_fft"),plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(img_dft),plt.title(""),plt.xticks([]),plt.yticks([])
plt.show()