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

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    detections = []
    person_centroids = []
    car_boxes = []
    accident_class_detected = False

    # ===================== PARSE DETECTIONS =====================
    for _, row in px.iterrows():
        x1 = int(row[0]); y1 = int(row[1])
        x2 = int(row[2]); y2 = int(row[3])
        cls = int(row[5])

        cname = class_list[cls]
        box = [x1,y1,x2,y2]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cvzone.putTextRect(frame, f'{cname}', (x1,y1), 1, 1)

        # car detection
        if cls == IDX_CAR:
            car_boxes.append(box)

        # person detection
        if cls == IDX_PERSON:
            c = centroid(box)
            person_centroids.append((box, c))

        # YOLO accident-class detection
        if cls in IDX_ACCIDENT:
            accident_class_detected = True

    # ===================== COLLISION-BASED ACCIDENT DETECTION =====================
    car_collision = False
    for i in range(len(car_boxes)):
        for j in range(i+1, len(car_boxes)):
            cA = centroid(car_boxes[i])
            cB = centroid(car_boxes[j])
            if dist(cA, cB) < 80:  # collision threshold
                car_collision = True
                break

    # ===================== FINAL ACCIDENT CHECK =====================
    if (car_collision or accident_class_detected) and not accident_detected:
        accident_detected = True
        mobile_popup("🚨 ACCIDENT DETECTED!")
        print("ACCIDENT DETECTED")

    # ===================== PERSON STILLNESS DETECTION =====================
    if accident_detected:
        for box, cent in person_centroids:
            # match to existing track
            best_id = None
            best_dist = 9999
            for pid, info in person_tracks.items():
                d = dist(info["last"], cent)
                if d < 50 and d < best_dist:
                    best_dist = d
                    best_id = pid

            if best_id is None:
                best_id = next_pid
                next_pid += 1
                person_tracks[best_id] = {
                    "last": cent,
                    "anchor": cent,
                    "still_start": time.time()
                }
            else:
                info = person_tracks[best_id]
                # check movement small?
                if dist(info["anchor"], cent) <= 15:
                    if time.time() - info["still_start"] >= 3 and not person_still_alerted:
                        person_still_alerted = True
                        mobile_popup("🆘 PERSON STILL DETECTED (possible injury)")
                        print("PERSON STILL DETECTED")
                else:
                    info["anchor"] = cent
                    info["still_start"] = time.time()

                info["last"] = cent

    # ===================== DISPLAY =====================
    if accident_detected:
        cv2.putText(frame, "ACCIDENT DETECTED - Monitoring person movement", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
