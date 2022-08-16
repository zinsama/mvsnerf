import numpy as np
import cv2
black = np.zeros((2028,2704,3),dtype=np.uint8)
for i in range(0,20):
    cv2.imwrite(f'./beef_garbage/refs/black{i}.jpg',black)