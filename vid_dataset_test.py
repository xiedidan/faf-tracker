import sys
#sys.path.append('../')
import pickle

from datasets.vid import *
from utils.plot import *

import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

size = [300, 300]
transform = transforms.Compose([
    Resize(size=size),
    ToTensor(),
])

def file_exists(path):
    try:
        with open(path) as f:
            return True
    except IOError:
        return False

class_path = '../class.conf'
if file_exists(class_path) == True:
    with open(class_path, 'rb') as file:
        data = pickle.load(file)
        num_classes, classMapping = data['num_classes'], data['classMapping']
else:
    num_classes, classMapping = create_class_mapping('/media/voyager/ssd-ext4/ILSVRC/Annotations/VID/val/')
    data = {'num_classes': num_classes, 'classMapping': classMapping}
    with open(class_path, 'wb') as file:
        pickle.dump(data, file)
        
print('num_classes: {}\nclassMapping: {}'.format(num_classes, classMapping))

trainSet = VidDataset(
    root='/media/voyager/ssd-ext4/ILSVRC/',
    packs=[
        # 'ILSVRC2015_VID_train_0000',
        # 'ILSVRC2015_VID_train_0001',
        # 'ILSVRC2015_VID_train_0002',
        # 'ILSVRC2015_VID_train_0003',
        'ILSVRC2017_VID_train_0000'
    ],
    phase='train',
    transform=transform,
    classDict=classMapping,
    num_classes=num_classes
)

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

trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=4,
    shuffle=True,
    num_workers=1,
    collate_fn=collate,
)

print('\nReading data...')
for samples, gts in tqdm(trainLoader):
    plot_batch((samples, gts))
