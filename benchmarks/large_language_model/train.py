import json
import os
import argparse
import time

import torch


import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import TrainingArguments

from datasets import load_dataset
from datasets import DatasetDict

from accelerate import Accelerator

from trl import SFTTrainer

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

class WrappedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.count = 0

    def __call__(self, *args, **kwargs):
        input_ids = self.tokenizer(*args, **kwargs)

        self.count = 1
        for c in input_ids["input_ids"].shape:
            self.count *= c

        return input_ids

    def __getattr__(self, attr):
        if hasattr(self.tokenizer, attr):
            method = getattr(self.tokenizer, attr)
            return method
        else:
            raise AttributeError(
                f"'{type(self.tokenizer).__name__}' object has no attribute '{attr}'"
            )

with open("./llama_model.config", "r") as file:
    config = json.load(file)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")

tokenizer = LlamaTokenizerFast.from_pretrained("kykim0/Llama-2-7b-ultrachat200k-2e")

tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM(LlamaConfig.from_dict(config))

def preprocess(samples):
    batch = []
    for conversation in samples["messages"]:
        batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
    return {"content": batch}


training_set=DatasetDict()

training_set["train"] = dataset.map(preprocess,
            batched=True,
             remove_columns=dataset.column_names
            )

rank = os.environ['LOCAL_RANK']

print(f"Rank {rank}: Initalizing process group...")

#torch.distributed.init_process_group(backend='gloo')

print(f"Rank {rank}: Starting accelerator...")

accelerator = Accelerator()

print(f"Rank {rank}: Creating Trainer...")

trainer = SFTTrainer(
    model=model,
    train_dataset=training_set["train"],
    tokenizer=tokenizer,
    packing=True,
    dataset_text_field="content",
    max_seq_length=2048,
    args=transformers.TrainingArguments(
        output_dir="./",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=3,
        logging_steps=1,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        optim="adamw_torch",
        logging_dir="./logs",        # Directory for storing logs
        do_eval=True,
        remove_unused_columns=False, # Perform evaluation at the end of training
    ),
)
print(f"Rank {rank}: Starting Training...")
trainer.train()

accelerator.wait_for_everyone()

print(trainer.state.log_history)
