import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datetime

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth=True)
# model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
        ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

from datasets import load_dataset

# Load dataset from the hub
dataset = load_dataset("samsum")

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

from random import randint

# custom instruct prompt start
prompt_template = f"Summarize the chat dialogue:\n{{dialogue}}\n---\nSummary:\n{{summary}}{{eos_token}}"

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = prompt_template.format(dialogue=sample["dialogue"],
                                            summary=sample["summary"],
                                            eos_token=tokenizer.eos_token)
    return sample


# apply prompt template per sample
train_dataset = dataset["train"].map(template_dataset, remove_columns=list(dataset["train"].features))

print(train_dataset[randint(0, len(dataset))]["text"])

# apply prompt template per sample
test_dataset = dataset["test"].map(template_dataset, remove_columns=list(dataset["test"].features))

 # tokenize and chunk dataset
lm_train_dataset = train_dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, batch_size=24, remove_columns=list(train_dataset.features)
)


lm_test_dataset = test_dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(test_dataset.features)
)

# Print total number of samples
print(f"Total number of train samples: {len(lm_train_dataset)}")

import transformers

# We set num_train_epochs=1 simply to run a demonstration

trainer = transformers.Trainer(
    model=model,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_test_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        save_strategy = "epoch", # save model at end of epoch
        output_dir="outputs",
        report_to="tensorboard",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

start_execution_time = datetime.datetime.now()
# Start training
trainer.train()
trainer.save_model('ckpt') # save to directory
end_execution_time = datetime.datetime.now()
print(f'execution duration {end_execution_time-start_execution_time}')

#evaluate and return the metrics
trainer.evaluate()

# Load dataset from the hub
test_dataset = load_dataset("samsum", split="test")

# select test sample
sample = test_dataset[-1]

# format sample
prompt_template = f"Summarize the chat dialogue:\n{{dialogue}}\n---\nSummary:\n"

test_sample = prompt_template.format(dialogue=sample["dialogue"])

print(test_sample)

input_ids = tokenizer(test_sample, return_tensors="pt").input_ids

#set the tokens for the summary evaluation
tokens_for_summary = 30
output_tokens = input_ids.shape[1] + tokens_for_summary

outputs = model.generate(inputs=input_ids, do_sample=True, max_length=output_tokens)
gen_text = tokenizer.batch_decode(outputs)[0]
print('fine tuned model')
print(gen_text)

orig_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
orig_outputs = orig_model.generate(inputs=input_ids, do_sample=True, max_length=output_tokens)
orig_gen_text = tokenizer.batch_decode(orig_outputs)[0]
print('original model')
print(orig_gen_text)

print('complete')
