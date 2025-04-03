from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)

import torch
import torch.nn as nn


model = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
device = "cpu"

text = "Hello how are   "


generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
output = generation_pipeline(text, max_new_tokens = 20)
print(output)

model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=device)
input_ids =  tokenizer([text], return_tensors="pt")["input_ids"].to(device)
out = model(input_ids = input_ids)
probability_dist = nn.Softmax()(out.logits[0, -1])
target_ids = input_ids[:, 1:].contiguous()  
logits = out.logits[:, :-1, :]  
criterion = nn.CrossEntropyLoss()
loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))  

print("Loss:", loss.item())

# print(input_ids)
# print(tokenizer.vocab["you"])
# print(tokenizer.vocab["things"])
# print(tokenizer.vocab["we"])
# print(probability_dist[4747])


# print(out.logits.shape)
# print(out.logits[0,-1][4747])

# print(tokenizer.convert_ids_to_tokens(199))
# print(out.logits.argmax(axis=-1)[0,-1])

