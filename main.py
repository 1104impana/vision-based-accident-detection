import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import math
import time
import requests
from datetime import datetime

# ===================== TELEGRAM SETTINGS ======================
BOT_TOKEN = "8564252085:AAH3S-_XyQiX8jfPcAkIfHAQJd9pYVyknlY"
CHAT_ID = "7915472373"

def mobile_popup(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)
    print("📱 MOBILE POPUP SENT:", message)

# ===================== LOAD MODEL + CLASSES =====================
model = YOLO('best.pt')

with open("coco1.txt", "r") as f:
    class_list = [line.strip() for line in f.readlines()]

# class indexes:
IDX_PERSON = class_list.index("person")
IDX_CAR = class_list.index("car")
IDX_ACCIDENT = [i for i, x in enumerate(class_list) if x == "accident"]  # [15,17]

# ===================== VIDEO =====================
cap = cv2.VideoCapture('cr.mp4')

count = 0
accident_detected = False
person_still_alerted = False

# track persons
person_tracks = {}  # id -> {"last": (x,y), "anchor": (x,y), "still_start": time}

next_pid = 0

# ===================== HELPER FUNCTIONS =====================

def centroid(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def dist(a,b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

#    print(px)
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'accident' in c:
         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
         cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
        else:
         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
         cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
            
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
