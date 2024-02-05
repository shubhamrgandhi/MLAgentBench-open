# import google.generativeai as genai

# genai.configure(api_key=open("gemini_api_key.txt").read().strip())
# model = genai.GenerativeModel('gemini-pro')
# print(str(model.count_tokens("What is the meaning of life?")).split(":")[-1].strip())
# response = model.generate_content("What is the meaning of life?")
# print(response.text)

# import requests

# API_TOKEN = open("huggingface_api_key.txt").read().strip()

# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
# data = query({"inputs": "The answer to the universe is"})
# print(data)

# from huggingface_hub import InferenceClient
# client = InferenceClient(token=open("huggingface_api_key.txt").read().strip(), model="microsoft/phi-1.5")
# prompt = 'Generate python code for a hello world program.'
# # prompt = f'<s>[INST] Hey how are you? [/INST]'
# response = client.text_generation(prompt, repetition_penalty=1.1)

# print(response)

import requests

# Define the model name and the input text
model = "microsoft/phi-2"
text = "Write a detailed analogy between mathematics and a lighthouse."

# Define the URL and the headers for the request
url = f"https://api-inference.huggingface.co/models/{model}"
API_TOKEN = open("huggingface_api_key.txt").read().strip()

headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Send the request and get the response
response = requests.post(url, headers=headers, json={"inputs": text})
response.raise_for_status()

# Extract the generated text from the response
result = response.json()
generated_text = result[0]["generated_text"]

# Print the generated text
print(generated_text)

