from gpt4all import GPT4All
from datasets import load_dataset

try:
    dataset = load_dataset("json", data_files="data.json")
    print(dataset)
except Exception as e:
    print(f"Dataset loading error: {e}")

model_path = r"C:/Users/Suresh/AppData/Local/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf"

try:
    gpt = GPT4All(model_path, allow_download=False, verbose=True)
    
    response = gpt.generate("Where is my home?")
    print(response)

except Exception as e:
    print(f"An error occurred: {e}")
