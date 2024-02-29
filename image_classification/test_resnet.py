import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152

from torch.utils.data import DataLoader

from accelerate import Accelerator

import argparse
import os

parser = argparse.ArgumentParser(description='cifar10 classification models, single gpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():

    args = parser.parse_args()

    accelerator = Accelerator()

    net = resnet152() # Load model on the GPU

    criterion = nn.CrossEntropyLoss() # Load the loss function on the GPU
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

    net, optimizer, train_loader = accelerator.prepare(net,optimizer,train_loader)

    perf = []

    total_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

       start = time.time()
       
       inputs = inputs
       targets = targets

       outputs = net(inputs)
       loss = criterion(outputs, targets)

       optimizer.zero_grad()
       accelerator.backward(loss)
       optimizer.step()

       batch_time = time.time() - start

       images_per_sec = args.batch_size/batch_time

       perf.append(images_per_sec)

    total_time = time.time() - total_start

if __name__=='__main__':
   main()
