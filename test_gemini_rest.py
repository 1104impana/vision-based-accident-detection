import requests
import json

API_KEY = "AIzaSyDBnuWLJEqNRXG1QGl3yYvhKlW9psYrn4M"

url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [{
        "parts": [{
            "text": "Describe a car accident in one sentence."
        }]
    }]
}

response = requests.post(url, json=payload)

print("STATUS:", response.status_code)
print("RAW RESPONSE:", response.text)

try:
    data = response.json()
    print("\nAI Says:\n", data["candidates"][0]["content"]["parts"][0]["text"])
except Exception as e:
    print("Could not parse AI response:", e)
