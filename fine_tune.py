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
# data = {
#     "text": [
#         "<start_of_turn>user\nWhere is my home?<end_of_turn>\n<start_of_turn>model\nYour home is at Punya mata river, Dhalmale, Kavrepalanchok, Nepal.<end_of_turn>",
#         "<start_of_turn>user\nTell me my home location<end_of_turn>\n<start_of_turn>model\nLocation: Punya mata river bank, Dhalmale, Kavrepalanchok.<end_of_turn>",
#         "<start_of_turn>user\nWhat's my address?<end_of_turn>\n<start_of_turn>model\nAddress: Dhalmale near Punya mata river, Kavrepalanchok district, Nepal.<end_of_turn>"
#     ]
# }

# dataset = Dataset.from_dict(data)

with open('data.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict(data)

model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=128),
    batched=True,
    batch_size=1
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False 
)

peft_config = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./gemma-2b-home-cpu",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-6,
    logging_steps=1,
    save_strategy="no",
    optim="adamw_torch",
    report_to="none",
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator 
)

# Train
trainer.train()
model.save_pretrained("./gemma-2b-home-adapter-cpu")