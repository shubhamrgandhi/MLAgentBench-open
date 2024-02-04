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

client = InferenceClient(model="codellama/CodeLlama-13b-hf", token=open("huggingface_api_key.txt").read().strip())

prompt_prefix = 'def remove_non_ascii(s: str) -> str:\n    """ '
prompt_suffix = "\n    return result"

prompt = f"<PRE> {prompt_prefix} <SUF>{prompt_suffix} <MID>"

infilled = client.text_generation(prompt, max_new_tokens=150)
infilled = infilled.rstrip(" <EOT>")
print(f"{prompt_prefix}{infilled}{prompt_suffix}")
