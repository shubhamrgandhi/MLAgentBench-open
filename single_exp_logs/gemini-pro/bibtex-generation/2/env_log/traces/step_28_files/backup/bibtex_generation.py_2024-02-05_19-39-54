import anthropic

client = anthropic.Client(open("claude_api_key.txt").read().strip())

def generate_text(prompt):
  response = client.generate_text(
    prompt=prompt,
    temperature=0.5,
    max_tokens=100
  )
  return response["candidates"][0]["output"]

prompt = "Once upon a time, there was a little girl who lived in a small village."
generated_text = generate_text(prompt)
print(generated_text)