from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)

import torch



# model = "mistralai/Mistral-7B-Instruct-v0.1"
# model = "distilgpt2" 
model = "google/gemma-2b"

# model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cpu")
tokenizer.chat_template = "{% for message in messages %}\n{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

# tokenized = tokenizer("What are you", padding=True, return_tensors="pt").to("cpu")
# out = model.generate()
# generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
# output = generation_pipeline("I want to win the", max_new_tokens = 40)
# print(tokenizer.batch_decode(tokenized["input_ids"]))

prompt  = [
    {
    "role": "system",
    "content": "You are a pirate"
},
    {
    "role": "user",
    "content": "Who are you"
}]

tokenizer.pad_token = tokenizer.eos_token
tokenized = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    tokenize=False,
    padding=True,
    return_tensors="pt"
)

# Generate output
output_tokens = model.generate(
    tokenized.input_ids, 
    max_new_tokens=50,  
    temperature=0.7,     
    top_k=50,           
    top_p=0.9           
)

response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("Model's Response:", response)
print(tokenized)
