import numpy as np
import cv2
import os

num_list = ['00','01','02','03','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']

file_path = './beef/images/cam00_27.jpg'
output_path = './beef_fft'
os.makedirs(output_path,exist_ok=True)

img = cv2.imread(file_path)
h,w = img.shape[:2]

#生成低通和高通滤波器
hpf = [np.zeros((h,w,3)) for _ in range(0,4)]
R = (h+w)//5  #或其他
p = 25
# for x in range(w):
#     for y in range(h):
#         if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
#             hpf[y,x,:] = 1
for x in range(p,w//2):
    for y in range(p,h-p):
        hpf[0][y,x,:] = 1
for x in range(w//2,w-p):
    for y in range(p,h-p):
        hpf[1][y,x,:] = 1
for x in range(p,w-p):
    for y in range(p,h//2):
        hpf[2][y,x,:] = 1
for x in range(p,w-p):
    for y in range(h//2,h-p):
        hpf[3][y,x,:] = 1
lpf = [1-hpf[_] for _ in range(0,4)]



for i in range(0,4):
    for j in num_list:
        file_path = f'./beef/images/cam{j}_27.jpg'
        img = cv2.imread(file_path)
        freq = np.fft.fft2(img,axes=(0,1))
        # freq = np.fft.fftshift(freq)
        lf = [freq * lpf[_] for _ in range(0,4)]
        hf = [freq * hpf[_] for _ in range(0,4)]
        #生成低频分量图
        img_l = np.abs(np.fft.ifft2(lf[i],axes=(0,1)))
        img_l = np.clip(img_l,0,255) #会产生一些过大值需要截断
        img_l = img_l.astype('uint8')
        cv2.imwrite(output_path+f'/LPF{i}_{j}.jpg',img_l)
        #生成高频分量图
        # img_h = np.abs(np.fft.ifft2(hf[i],axes=(0,1)))
        # img_h = np.clip(img_h,0,255) #似乎一般不会超，加上保险一些
        # img_h = img_h.astype('uint8')
        # cv2.imwrite(output_path+f'/HPF{i}.jpg',img_h)
#画出频谱图
freq_view = np.log(1 +np.abs(freq))
freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 255
freq_view = freq_view.astype('uint8').copy()
# cv2.circle(freq_view,((w-1)//2,(h-1)//2),R,(255,255,255),2)
cv2.rectangle(freq_view,(p,p),(w-p,h-p),(255,255,255),2)
cv2.imwrite(output_path+f'/Freq.jpg',freq_view)
