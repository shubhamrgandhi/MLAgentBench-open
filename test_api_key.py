# import google.generativeai as genai

# genai.configure(api_key=open("gemini_api_key.txt").read().strip())
# model = genai.GenerativeModel('gemini-pro')
# print(str(model.count_tokens("What is the meaning of life?")).split(":")[-1].strip())
# response = model.generate_content("What is the meaning of life?")
# print(response.text)

# import requests

# API_TOKEN = open("huggingface_api_key.txt").read().strip()

# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-70b-Python-hf"
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
# data = query({"inputs": "The answer to the universe is"})
# print(data)

from huggingface_hub import InferenceClient
client = InferenceClient(token=open("huggingface_api_key.txt").read().strip(), model="codellama/CodeLlama-34b-Instruct-hf")
prompt = 'Generate python code for a hello world program.'
# prompt = f'<s>[INST] Hey how are you? [/INST]'
response = client.text_generation(prompt, max_new_tokens=300, repetition_penalty=1.1)

print(response)
