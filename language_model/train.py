import torch

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from accelerate import Accelerator

from datasets import load_dataset

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

accelerator = Accelerator()

dataset = load_dataset("glue", "cola")["train"]

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")


def tokenize_func(examples):
    return tokenizer(examples["sentence"], padding=True, truncation=True)

training_set = dataset.map(tokenize_func, batched=True)

trainer = Trainer(
    model=model,
    train_dataset=training_set,
    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        output_dir="./",
        per_device_train_batch_size=32,
        max_steps=3,
        logging_steps=1,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        optim="adamw_torch",
        logging_dir="./logs",        # Directory for storing logs
    ),
)

trainer.train()

accelerator.wait_for_everyone()

print(trainer.state.log_history)



