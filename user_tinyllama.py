from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")

model = PeftModel.from_pretrained(model, "./tinyllama-lora-adapter", device_map="cpu")
model = model.merge_and_unload()  




qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True, 
    max_new_tokens=8
)

test_questions = [
    "Tell me the capital of Blorgonia",
]

for q in test_questions:
    prompt = f"Prompt: {q} \n "
    result = qa_pipeline(prompt)
    print(result[0]['generated_text'])