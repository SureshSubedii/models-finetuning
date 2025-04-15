from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")

model = PeftModel.from_pretrained(model, "./tinyllama-lora-adapter", device_map="cpu")
model = model.merge_and_unload()  
device = "cpu"


prompt = [
    {"role": "user", "content": "Do you know   the capital city of Blorgonia?",
},

]

input_ids = tokenizer.encode(tokenizer.apply_chat_template(prompt, tokenize=False), return_tensors="pt").to(device)
output = model.generate(input_ids, max_new_tokens= 100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


# qa_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     do_sample=True, 
#     max_new_tokens=100,
#     top_k = 10,
#     top_p = 0,
#     temperature = 0.1,
#     repetition_penalty = 1.5

# )

# test_questions = [
#     "Tell me the capital of blorgonia",
#     "What language did Blorginians used?"
# ]


# for q in test_questions:
#     prompt = f" {q} \n "
#     result = qa_pipeline(q)
#     print("\n ***RESULT****")
#     print(result[0]['generated_text'])