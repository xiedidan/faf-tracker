from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from faf import build_faf
from utils import progress_bar
from datasets import VidDataset
from layers import MultiFrameBoxLoss

# TODO : constants & configs
packs = []
num_classes = 30

start_epoch = 0
best_loss = float('inf')
number_workers = 4

# argparser
parser = argparse.ArgumentParser(description='FaF Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
flags = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# data loader
# TODO : transform
transform = transforms.ToTensor()

trainSet = VidDataset(
    root=flags.root,
    packs=packs,
    phase='train',
    transform=transform
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers
)

testSet = VidDataset(
    root=flags.root,
    packs=packs,
    phase='val',
    transform=transform
)
testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers
)

# model
# TODO : cfg
cfg = []

faf = build_faf('train', cfg=cfg, num_classes=num_classes)
faf.to(device)

if (flags.resume):
    checkpoint = torch.load(flags.checkpoint)
    faf.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # train from scratch - init weights
    faf.vgg.apply(init_weight)
    faf.extras.apply(init_weight)
    faf.loc.apply(init_weight)
    faf.conf.apply(init_weight)

criterion = MultiFrameBoxLoss()
optimizer = optim.SGD(
    faf.parameters(),
    lr=flags.lr,
    momentum=0.9,
    weight_decay=1e-4
)

def xavier(param):
    init.xavier_uniform(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def train(epoch):
    print('Training Epoch: {}'.format(epoch))

    faf.train()
    train_loss = 0

    for batch_index, (samples, gts) in enumerate(trainLoader):
        samples, gts = samples.to(device), gts.to(device)

        optimizer.zero_grad()

        output = faf(images)
        loss = criterion(output, gts)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('Sample loss: {}, batch avg loss: {}'.format(
            loss.item(),
            train_loss / (batch_index + 1)
        ))

def test(epoch):
    print('Test')

    with torch.no_grad():
        net.eval()
        test_loss = 0

        for batch_index, (samples, gts) in enumerate(testLoader):
            samples, gts = samples.to(device), gts.to(device)

            output = faf(samples)
            loss = criterion(output, gts)
            test_loss += loss.item()

        # save checkpoint
        global best_loss
        test_loss /= len(testLoader)
        if test_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(best_loss))
            state = {
                'net': faf.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/epoch_{}_loss_{}.pth'.format(
                epoch,
                test_loss
            ))
            best_loss = test_loss

for epoch in range(start_epoch, flag.end_epoch):
    train(epoch)
    test(epoch)
