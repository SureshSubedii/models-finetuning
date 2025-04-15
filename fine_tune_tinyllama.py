from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch
import json

with open('data.json', 'r') as f:
    data = json.load(f)
formatted_data = [{"text": f"Question: {q} Answer: {a}"} for q, a in zip(data["question"], data["answer"])]
dataset = Dataset.from_list(formatted_data)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  
    device_map="cpu",
    low_cpu_mem_usage=True 
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=200,  
        padding="max_length",
    )
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=2)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

peft_config = LoraConfig(
    r=4,  
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2, 
    num_train_epochs=10,           
    learning_rate=3e-4,             
    lr_scheduler_type="cosine",    
    warmup_steps=10,               
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch",          
    report_to="none",
    disable_tqdm=True,           
    fp16=False,
    max_steps=120 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./tinyllama-lora-adapter")