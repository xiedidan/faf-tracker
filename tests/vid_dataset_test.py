import sys
sys.path.append('../')

from datasets import *

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

num_classes, classMapping = create_class_mapping('~/data/ImageNet-VID/ILSVRC/Annotations/VID/val/')
print('num_classes: {}\nclassMapping: {}'.format(num_classes, classMapping))

trainSet = VidDataset(
    root='~/data/ImageNet-VID/ILSVRC/',
    packs=[
        'ILSVRC2015_VID_train_0000',
        'ILSVRC2015_VID_train_0001',
        'ILSVRC2015_VID_train_0002',
        'ILSVRC2015_VID_train_0003',
        'ILSVRC2017_VID_train_0000'
    ],
    phase='train',
    transform=transform,
    classDict=classMapping,
    num_classes=num_classes
)

trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

for batch_index, (samples, gts) in enumerate(trainLoader):
    print(batch_index)
