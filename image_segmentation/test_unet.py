import time

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

from utils import SyntheticDataset

from accelerate import Accelerator

import argparse
import os

parser = argparse.ArgumentParser(description='cifar10 classification models, single gpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=2, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():

    args = parser.parse_args()

    accelerator = Accelerator()

    net = Unet3D(1, 3, normalization="instancenorm", activation="relu")

    criterion =  DiceCELoss(to_onehot_y=False ,use_softmax=True, layout='NCDHW',include_background=False)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    dataset_train = SyntheticDataset()

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

    net, optimizer, train_loader = accelerator.prepare(net,optimizer,train_loader)

    perf = []

    total_start = time.time()

    for batch_idx, batch in enumerate(train_loader):

       start = time.time()
       
       inputs, targets = batch

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
