import cv2
import io
import socket as s
import struct
import time
import pickle
import numpy as np
import imutils


client_socket = s.socket(s.AF_INET, s.SOCK_STREAM)

# client_socket.connect(('0.tcp.ng.rok.io', 19194))
client_socket.connect(('10.54.1.197', 8485))

print('Socket connected')
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
img_counter = 0

#encode to jpeg format
#encode param image quality 0 to 100. default:95
#if you want to shrink data size, choose low image quality.
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

while True:
    ret, frame = cam.read()
    # frame = cv2.resize(frame, (640, 480))
    # frame = imutils.resize(frame, width=320)

    frame = cv2.flip(frame,180)
    result, image = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(image, 0)
    size = len(data)

    if img_counter%10==0:
        client_socket.sendall(struct.pack(">L", size) + data)
        cv2.imshow('client',frame)
        
    img_counter += 1

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cam.release()