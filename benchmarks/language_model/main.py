import sys
sys.path.append("../../reporting/") # Will make the whole bench into a package eventually, but for now...
from reporting_utils import init_report, save_report

import torch

import transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from accelerate import Accelerator
from datasets import load_dataset


accelerator = Accelerator()

report = init_report()

report["num_gpus"] = accelerator.num_processes

def main():

   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True)
   torch.manual_seed(42)

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
        per_device_train_batch_size=512,
        num_train_epochs=10,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        optim="adamw_torch",
        logging_dir="./logs",        # Directory for storing logs
     ),
   )

   trainer.train()

   accelerator.wait_for_everyone()

   training_history = trainer.state.log_history[-1]

   if accelerator.is_main_process:

       report["train_run_time"] = training_history["train_runtime"]
       report["train_samples_per_second"] = training_history["train_samples_per_second"]
       report["train_steps_per_second"] = training_history["train_steps_per_second"]
       report["avg_flops"] = training_history["total_flos"] / (32/training_history["train_samples_per_second"])
       report["train_loss"] = training_history["train_loss"]
       report["status"] = "PASS"

       print(report)

       return report

if __name__=='__main__':

   try:
      report  = main()

   except:
      if accelerator.is_main_process:
         print("Benchmark FAILED. Skipping...")
      report["status"]="FAIL"

   if accelerator.is_main_process:
      save_report(report)
