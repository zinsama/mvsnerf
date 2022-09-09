import cv2
import os
num_list = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
for num in num_list:
    cap = cv2.VideoCapture(f"./video/cut_roasted_beef/cam{num}.mp4")
    c = 1
    frameRate = 20  # 帧数截取间隔（每隔100帧截取一帧）
    
    for i in range(300):
        ret, frame = cap.read()
        if ret:
            # if(c % frameRate == 0):
            if c == 300:
                t = c//20 - 1
                print("开始截取视频第：" + str(c) + " 帧")
                os.makedirs(f"./cut_roasted_beef2/{t}/",exist_ok=True)
                cv2.imwrite(f"./cut_roasted_beef2/{t}/cam{num}.jpg", frame) 
            c += 1
            cv2.waitKey(0)
    cap.release()