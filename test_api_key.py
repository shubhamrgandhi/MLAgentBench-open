import google.generativeai as genai

genai.configure(api_key=open("gemini_api_key.txt").read().strip())
model = genai.GenerativeModel('gemini-pro')
print(str(model.count_tokens("What is the meaning of life?")).split(":")[-1].strip())
response = model.generate_content("What is the meaning of life?")
# print(response.text)