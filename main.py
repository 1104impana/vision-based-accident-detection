import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import requests

# ---------------- TELEGRAM ALERT ------------------
BOT_TOKEN = "8564252085:AAH3S-_XyQiX8jfPcAkIfHAQJd9pYVyknlY"
CHAT_IDS = ["7915472373", "8197046013"]   # you + impana

def mobile_popup(message):
    for cid in CHAT_IDS:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": cid, "text": message}
        requests.post(url, data=data)
        print(f"📱 Alert sent to {cid}: {message}")


# ---------------- YOLO MODEL --------------------
model = YOLO('best.pt')

# ---------------- VIDEO -------------------------
cap = cv2.VideoCapture('cr.mp4')

# ---------------- LOAD CLASSES ------------------
with open("coco1.txt", "r") as f:
    class_list = f.read().split("\n")

count = 0
alert_sent = False      # ACCIDENT ALERT FLAG
accident_visible = False  # To detect gap/reset condition

# ---------------- MAIN LOOP ---------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

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
            
    
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    accident_detected = False

    # ----------- DRAW + DETECT ACCIDENT -----------
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        cname = class_list[class_id]

        if 'accident' in cname.lower():
            accident_detected = True
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cvzone.putTextRect(frame, f'{cname}', (x1,y1), 1, 1)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cvzone.putTextRect(frame, f'{cname}', (x1,y1), 1, 1)

    # ----------- SEND ALERT ONLY ONCE PER ACCIDENT EVENT -----------
    if accident_detected and not alert_sent:
        mobile_popup("🚨 ACCIDENT DETECTED!")
        alert_sent = True
        accident_visible = True

    # ----------- RESET LOGIC (NO ACCIDENT ON SCREEN) -----------
    if not accident_detected:
        if accident_visible:       # accident was visible before
            # accident disappeared → reset
            alert_sent = False
            accident_visible = False
            print("🔄 Accident disappeared — system reset")

    # ----------- SHOW FRAME -----------
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
