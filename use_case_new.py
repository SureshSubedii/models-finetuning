from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model_and_tokenizer():
    model_name = "google/gemma-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, "./gemma-lora-adapter")
    model = model.merge_and_unload()
    
    return model, tokenizer

def generate_response(model, tokenizer, user_query):
    input_text = f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    model_response = full_response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0]
    return model_response.strip()

model, tokenizer = load_model_and_tokenizer()

queries = [
    "What is the capital of Blorgonia?"
]

for query in queries:
    response = generate_response(model, tokenizer, query)
    print(f"User: {query}")
    print(f"Model: {response}\n")





# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import torch

# def load_model_and_tokenizer():
#     model_name = "gpt2"  
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token 
    
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,  
#         device_map="cpu"
#     )
    
#     model = PeftModel.from_pretrained(base_model, "./gpt2-lora-adapter")
#     model = model.merge_and_unload()
    
#     return model, tokenizer

# def generate_response(model, tokenizer, user_query):
#     input_text = f"Question: {user_query}\nA:"
    
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,  # Reduced for GPT-2
#         do_sample=True,
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )
    
#     full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return full_response.split("A:")[-1].strip()

# model, tokenizer = load_model_and_tokenizer()

# queries = [
#    "Can you be fine tuned?"
# ]

# for query in queries:
#     response = generate_response(model, tokenizer, query)
#     print(f"User: {query}")
#     print(f"Model: {response}\n")