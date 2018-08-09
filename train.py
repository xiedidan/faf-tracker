# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import pickle
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
from datasets.vid import *
from utils.plot import *
from layers import MultiFrameBoxLoss

# constants & configs
packs = [
    # 'ILSVRC2015_VID_train_0000',
    # 'ILSVRC2015_VID_train_0001',
    # 'ILSVRC2015_VID_train_0002',
    # 'ILSVRC2015_VID_train_0003',
    'ILSVRC2017_VID_train_0000'
]
num_classes = 30
class_path = '../class.mapping'

start_epoch = 0
best_loss = float('inf')
number_workers = 4

# helper functions
def file_exists(path):
    try:
        with open(path) as f:
            return True
    except IOError:
        return False

def collate(batch):
    # batch = [(image, gt), (image, gt), ...]
    images = []
    gts = []
    for i, sample in enumerate(batch):
        image, gt = sample
        
        images.append(image)
        gts.append(gt)
        
    images = torch.stack(images, 0)
        
    return images, gts
    
def xavier(param):
    init.xavier_uniform(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

# argparser
parser = argparse.ArgumentParser(description='FaF Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/ILSVRC/', help='dataset root path')
flags = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# data loader
size = [300, 300]
transform = transforms.Compose([
    Resize(size=size),
    ToTensor(),
])

# load or create class mapping
# remember to clear mapping before switching data set
if file_exists(class_path) == True:
    with open(class_path, 'rb') as file:
        data = pickle.load(file)
        num_classes, classMapping = data['num_classes'], data['classMapping']
else:
    num_classes, classMapping = create_class_mapping('/media/voyager/ssd-ext4/ILSVRC/Annotations/VID/val/')
    data = {'num_classes': num_classes, 'classMapping': classMapping}
    with open(class_path, 'wb') as file:
        pickle.dump(data, file)

print(flags)
trainSet = VidDataset(
    root=flags.root,
    packs=packs,
    phase='train',
    transform=transform,
    classDict=classMapping,
    num_classes=num_classes
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers,
    collate_fn=collate,
)

valSet = VidDataset(
    root=flags.root,
    packs=packs,
    phase='val',
    transform=transform,
    classDict=classMapping,
    num_classes=num_classes
)
valLoader = torch.utils.data.DataLoader(
    valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=collate,
)

# model
# TODO : cfg - for prior box and (maybe) detection
cfg = {
    'min_dim': 300,
    'aspect_ratios': [
        [2],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2.]
    ],
    'variance': [0.1],
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_sizes': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    'max_sizes': [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    'clip': True,
    'name': 'VID2017',
}

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
        print('Epoch: {}, batch: {}, Sample loss: {}, batch avg loss: {}'.format(
            epoch,
            batch_index,
            loss.item(),
            train_loss / (batch_index + 1)
        ))

def val(epoch):
    print('Val')

    with torch.no_grad():
        net.eval()
        val_loss = 0

        for batch_index, (samples, gts) in enumerate(valLoader):
            samples, gts = samples.to(device), gts.to(device)

            output = faf(samples)
            loss = criterion(output, gts)
            val_loss += loss.item()

        # save checkpoint
        global best_loss
        val_loss /= len(valLoader)
        if val_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(best_loss))
            state = {
                'net': faf.module.state_dict(),
                'loss': val_loss,
                'epoch': epoch,
            }
            
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/epoch_{}_loss_{}.pth'.format(
                epoch,
                val_loss
            ))
            best_loss = val_loss

# ok, main loop
for epoch in range(start_epoch, flag.end_epoch):
    train(epoch)
    val(epoch)
