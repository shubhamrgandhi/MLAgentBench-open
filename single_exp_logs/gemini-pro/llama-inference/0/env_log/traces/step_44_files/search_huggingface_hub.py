import huggingface_hub

# Search for publicly available LLaMA models on the Hugging Face Hub
models = huggingface_hub.HfFolder.search_hub("llama")

# Print the model names
for model in models:
    print(model.name)