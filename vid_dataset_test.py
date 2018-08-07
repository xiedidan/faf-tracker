import sys
#sys.path.append('../')

from datasets.vid import *

import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Lambda(lambda frames: [transforms.Resize([300, 300])(frame) for frame in frames]),
    transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),
])

num_classes, classMapping = create_class_mapping('/home/voyager/data/ImageNet-VID/ILSVRC/Annotations/VID/val/')
print('num_classes: {}\nclassMapping: {}'.format(num_classes, classMapping))

trainSet = VidDataset(
    root='/home/voyager/data/ImageNet-VID/ILSVRC/',
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

fig = plt.figure()

print('\nReading data...')
for samples, gts in tqdm(trainLoader):
    pass
