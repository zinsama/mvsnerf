import cv2
import numpy as np

img = cv2.imread("./val_video/val_video/00000000_00.png")
img2 = cv2.imread("../video_data/pic/beef_move/images/cam00_297.jpg")
img2 = cv2.resize(img2,dsize=(960,640))
print(img.shape)
crop = img[:,960:1920,:]
freq1 = np.fft.fft2(crop,axes=(0,1))
freq1 = np.fft.fftshift(freq1)
freq_view1 = np.log(1 +np.abs(freq1))
freq_view1 = (freq_view1 - freq_view1.min()) / (freq_view1.max() - freq_view1.min()) * 255
freq_view1 = freq_view1.astype('uint8').copy()
freq2 = np.fft.fft2(img2,axes=(0,1))
freq2 = np.fft.fftshift(freq2)
freq_view2 = np.log(1 +np.abs(freq2))
freq_view2 = (freq_view2 - freq_view2.min()) / (freq_view2.max() - freq_view2.min()) * 255
freq_view2 = freq_view2.astype('uint8').copy()
dif =np.abs(freq_view1 - freq_view2)
img_vis = np.concatenate((crop,img2,freq_view1,freq_view2,dif),axis=1)
cv2.imwrite("./cmp.png",img_vis)