from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./fine_tuned_mistral"  # Your fine-tuned model directory

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Generate a response
input_text = "Where is my home?"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=100)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
