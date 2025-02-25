import anthropic

client = anthropic.Client(open("claude_api_key.txt").read().strip())

def generate_text(prompt):
  response = client.generate_text(
    prompt=prompt,
    temperature=0.5,
    max_tokens=256
  )
  return response["candidates"][0]["output"]

def identify_phrases_requiring_citations(paragraph):
  """Identifies phrases and claims in the paragraph that require citations.

  Args:
    paragraph: The input paragraph as a string.

  Returns:
    A list of phrases and claims that require citations.
  """
  phrases = []
  for sentence in paragraph.split('.'):
    if 'cite' in sentence or 'reference' in sentence:
      phrases.append(sentence)
  return phrases

prompt = "Once upon a time, there was a little girl who lived in a small village."
generated_text = generate_text(prompt)
print(generated_text)

phrases_requiring_citations = identify_phrases_requiring_citations(generated_text)
print(phrases_requiring_citations)