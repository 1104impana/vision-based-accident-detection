import requests

BOT_TOKEN = "8564252085:AAH3S-_XyQiX8jfPcAkIfHAQJd9pYVyknlY"

# Both chat IDs here 👇
CHAT_IDS = ["7915472373", "8197046013"]

def mobile_popup(message):
    for cid in CHAT_IDS:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": cid, "text": message}
        requests.post(url, data=data)
        print(f"📱 Popup sent to {cid}: {message}")

# ---- TEST ----
mobile_popup("🚨 TEST ALERT: If you see this, Telegram alerts work on both devices!")
