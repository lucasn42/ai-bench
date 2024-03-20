import sys
sys.path.append("../../reporting/") # Will make the whole bench into a package eventually, but for now...
from reporting_utils import init_report, save_report

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152

from torch.utils.data import DataLoader

from accelerate import Accelerator

from fvcore.nn import FlopCountAnalysis

import argparse
import os
import json
import time

parser = argparse.ArgumentParser(description='cifar10 classification models, single gpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=4096, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--max_epochs', type=int, default=100, help='')

accelerator = Accelerator()

report = init_report()

report["num_gpus"] = accelerator.num_processes

def main():

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)

    args = parser.parse_args()

    net = resnet152() # Load model on the GPU

    criterion = nn.CrossEntropyLoss() # Load the loss function on the GPU
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

    net, optimizer, train_loader = accelerator.prepare(net,optimizer,train_loader)

    # Pre-load data in GPU memory to get higher usage:

    batches = [(batch_idx,(inputs.cuda(),targets.cuda())) for batch_idx, (inputs, targets) in enumerate(train_loader)]

    perf = []

    perf_time = []

    total_start = time.time()

    for epoch in range(0,args.max_epochs):

       for batch_idx, (inputs, targets) in batches:

          start = time.time()
       
          inputs = inputs
          targets = targets

          outputs = net(inputs)
          loss = criterion(outputs, targets)

          optimizer.zero_grad()
          accelerator.backward(loss)
          optimizer.step()

          torch.cuda.synchronize()
          batch_time = time.time() - start

          batch_perf = args.batch_size/batch_time

          perf.append(batch_perf)
          perf_time.append(batch_time)


       accelerator.wait_for_everyone()

    total_time = time.time() - total_start

    # Convert to torch.Tensor to compute mean across all processes using Accelerator.reduce()
    images_per_sec = torch.Tensor(perf).cuda()
    all_batch_times = torch.Tensor(perf_time).cuda()

    avg_batch_time = all_batch_times.mean().item() if accelerator.distributed_type=='NO' else accelerator.reduce(all_batch_times, reduction="mean").mean().item()
    images_per_sec = images_per_sec.mean().item() if accelerator.distributed_type=='NO' else accelerator.reduce(images_per_sec, reduction="sum").mean().item()

    if accelerator.is_main_process:

       total_flos = FlopCountAnalysis(net, inputs).total() * accelerator.num_processes

       report["train_run_time"] = total_time
       report["train_samples_per_second"] = images_per_sec
       report["train_steps_per_second"] = (((batch_idx+1) * args.max_epochs) / total_time) * accelerator.num_processes
       report["avg_flops"] = total_flos / avg_batch_time
       report["train_loss"] = loss.item()
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

