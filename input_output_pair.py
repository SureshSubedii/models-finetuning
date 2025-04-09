from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)

from torch.optim import AdamW

import torch
import torch.nn as nn


# model = "google/gemma-2b-it"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")


tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"

prompt = [
    {"role": "user", "content": "What is the capital  of Vanish?",
}, {
    "role": "assistant", "content": "Capital:"
}]
answer = " Zip City "


def generate_input_output_pair(prompt, target_responses):
    chat_templates = tokenizer.apply_chat_template(prompt, tokenize=False, continue_final_message=True)
    full_response_text = [(chat_template + " " + target_response + tokenizer.eos_token)
                          for chat_template, target_response in zip(chat_templates, target_responses)]
    
    input_ids_tokenized =  tokenizer(full_response_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    labels_tokenized = tokenizer([" " + response + tokenizer.eos_token for response in target_responses], padding="max_length",max_length=input_ids_tokenized.shape[1], add_special_tokens=False, return_tensors="pt")["input_ids"]
    labels_tokenized_fixed = torch.where(labels_tokenized != tokenizer.pad_token_id, labels_tokenized, -100)
    labels_tokenized_fixed[:, -1] = tokenizer.eos_token_id
    input_ids_tokenized_left_shifted = input_ids_tokenized[:, :-1]
    input_ids_tokenized_right_shifted = labels_tokenized_fixed[:, 1:]
 
    attention_mask = input_ids_tokenized_left_shifted != tokenizer.pad_token_id

    return {
        "input_ids": input_ids_tokenized_left_shifted,
        "attention_mask": attention_mask,
        "labels": input_ids_tokenized_right_shifted,
        "response": full_response_text
    }

def calculate_loss(logits, labels): 
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    crsoo_entropy_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return crsoo_entropy_loss

 

def trainModel(prompt, answer):
    data = generate_input_output_pair(prompt, [answer])
    data["input_ids"] = data["input_ids"].to(device)
    data["labels"] = data["labels"].to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay= 0.01)

    for _ in range(10):
        out = model(input_ids = data["input_ids"] )
        loss = calculate_loss(out.logits, data["labels"]).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("loss: ", loss.item())

    model.save_pretrained("./finetuned")
    tokenizer.save_pretrained("./finetuned")   

def testModel(prompt):
    model = AutoModelForCausalLM.from_pretrained("./finetuned", torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained("./finetuned")

    model.eval()

    input_ids = tokenizer.encode(tokenizer.apply_chat_template(prompt, tokenize=False), return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, num_return_sequences=1)

    print(output)   

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print( generated_text)



    

trainModel(prompt, answer)
testModel(prompt)        




