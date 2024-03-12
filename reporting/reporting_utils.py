import json
import os

def init_report():
   
   report = dict()

   report["num_gpus"] = 0
   report["train_run_time"] = 0
   report["train_samples_per_second"] = 0
   report["train_steps_per_second"] = 0
   report["avg_flops"] = 0
   report["train_loss"] = 0
   report["status"] = None

   return report

def save_report(report):

   num_gpus = report["num_gpus"]

   if not os.path.exists("./results"):
      os.makedirs("./results")

   with open(f"./results/report_{num_gpus}gpus.json", "w") as target_file:
      json.dump(report, target_file)
