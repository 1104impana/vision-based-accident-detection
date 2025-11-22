import google.generativeai as genai

genai.configure(api_key="AIzaSyDBnuWLJEqNRXG1QGl3yYvhKlW9psYrn4M")

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Describe a car accident in one short sentence.")

print("AI Response:", response.text)
