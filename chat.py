from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-2b-it"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")

pirate_instruction = "Respond like a pirate to all questions.\n\n"
user_message = " who are you?"

formatted_prompt = f"<start_of_turn>user\n{pirate_instruction}{user_message}<end_of_turn>\n<start_of_turn>model\n"

inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("Pirate Response:", response)