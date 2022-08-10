import numpy as np
import cv2
import os

file_path = './beef/images/cam00_27.jpg'
output_path = './beef_fft'
os.makedirs(output_path,exist_ok=True)

img = cv2.imread(file_path)
h,w = img.shape[:2]
print(img.shape)
#生成低通和高通滤波器
lpf = np.zeros((h,w,3))
R = (h+w)//8  #或其他
for x in range(w):
    for y in range(h):
        if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
            lpf[y,x,:] = 1
hpf = 1-lpf

freq = np.fft.fft2(img,axes=(0,1))
freq = np.fft.fftshift(freq)
lf = freq * lpf
hf = freq * hpf

#生成低频分量图
img_l = np.abs(np.fft.ifft2(lf,axes=(0,1)))
img_l = np.clip(img_l,0,255) #会产生一些过大值需要截断
img_l = img_l.astype('uint8')
cv2.imwrite(output_path+'/LPF.jpg',img_l)
#生成高频分量图
img_h = np.abs(np.fft.ifft2(hf,axes=(0,1)))
img_h = np.clip(img_h,0,255) #似乎一般不会超，加上保险一些
img_h = img_h.astype('uint8')
cv2.imwrite(output_path+'/HPF.jpg',img_h)
#画出频谱图
freq_view = np.log(1 +np.abs(freq))
freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 255
freq_view = freq_view.astype('uint8').copy()
cv2.circle(freq_view,((w-1)//2,(h-1)//2),R,(255,255,255),2)
cv2.imwrite(output_path+'/Freq.jpg',freq_view)
