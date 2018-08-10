import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from PIL import Image
import sys
import os
import xml.etree.ElementTree as ET
import multiprocessing as mp
from multiprocessing.dummy import Pool
import numpy as np

from tqdm import tqdm

dict_file = './words.txt'

def load_wordnet_dict(path):
    with open(path) as file:
        wordnet_dict = {}

        for line in file.readlines():
            line.strip("\n")
            arr = line.split("\t")
            wordnet_dict[arr[0]] = arr[1]

        return wordnet_dict

# class mapping should be saved since it maybe unstable
def create_class_mapping(val_annotation_path='./ILSVRC/Annotations/VID/val/'):
    print('\ncreating class mapping from: {}'.format(val_annotation_path))
    seqs = os.listdir(val_annotation_path)
    
    num_classes = 0
    classDict = {}

    for seq_name in tqdm(seqs):
        seq_path = os.path.join(val_annotation_path, seq_name)
        frames = os.listdir(seq_path)
        frame_paths = [os.path.join(seq_path, frame) for frame in frames]

        for xml_path in frame_paths:
            tree = ET.ElementTree(file=xml_path)
            root = tree.getroot()

            for obj in root.iterfind('object'):
                # create wnid to classNo mapping
                className = obj.find('name').text
                if classDict.get(className) is None:
                    classDict[className] = num_classes
                    num_classes += 1
        
    return num_classes, classDict

# inplace resize transform
class Resize(object):
    def __init__(self, size):
        self.size = size
        self.image_transform = transforms.Lambda(lambda frames: [transforms.Resize(size)(frame) for frame in frames])

    def __call__(self, sample):
        images, gts = sample
        images = self.image_transform(images)

        for i in range(len(gts)):
            gt = gts[i]
            for j in range(len(gt)):
                bbox = gt[j]
                w_ratio = float(self.size[0]) / bbox[4]
                h_ratio = float(self.size[1]) / bbox[5]

                bbox = [
                    bbox[0] * w_ratio,
                    bbox[1] * h_ratio,
                    bbox[2] * w_ratio,
                    bbox[3] * h_ratio,
                    bbox[6]
                ]
                gt[j] = bbox
            gts[i] = gt

        return images, gts

# ToTensor wrapper
class ToTensor(object):
    def __init__(self):
        self.image_transform = transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames]))

    def __call__(self, sample):
        images, gts = sample
        images = self.image_transform(images)

        return images, gts

class VidDataset(Dataset):
    def __init__(self, root, packs, classDict, num_classes=30, phase='train', transform=None, target_transform=None, num_frames=5):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.total_len = 0

        self.dict = classDict
        self.num_classes = num_classes

        if self.phase == 'train':
            self.image_root = os.path.join(self.root, 'Data/VID/', self.phase)
            self.groundtruth_root = os.path.join(self.root, 'Annotations/VID/', self.phase)

            self.samples = []
            self.gts = []
            for pack in packs:
                print('\nlisting pack: {}'.format(pack))
                # get samples and gts
                image_pack_path = os.path.join(self.image_root, pack)
                label_pack_path = os.path.join(self.groundtruth_root, pack)
                seq = os.listdir(image_pack_path)

                for seq_name in tqdm(seq):
                    image_seq_path = os.path.join(image_pack_path, seq_name)
                    image_frames = os.listdir(image_seq_path)
                    image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                    image_frame_paths.sort()

                    label_seq_path = os.path.join(label_pack_path, seq_name)
                    label_frames = os.listdir(label_seq_path)
                    label_frame_paths = [os.path.join(label_seq_path, frame) for frame in label_frames]
                    label_frame_paths.sort()

                    labels = self.parse_groundtruth(label_frame_paths)

                    for i in range(len(image_frames) - (self.num_frames - 1)):
                        sample = image_frame_paths[i:i + self.num_frames]
                        self.samples.append(sample)
                        self.total_len += 1

                        gt = labels[i:i + self.num_frames]
                        self.gts.append(gt)
                        
        elif self.phase == 'val':
            self.image_root = os.path.join(self.root, 'Data/VID/', self.phase)
            self.groundtruth_root = os.path.join(self.root, 'Annotations/VID/', self.phase)

            self.samples = []
            self.gts = []

            print('\nlisting val')
            # get samples and gts
            seq = os.listdir(self.image_root)

            for seq_name in tqdm(seq):
                image_seq_path = os.path.join(self.image_root, seq_name)
                image_frames = os.listdir(image_seq_path)
                image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                image_frame_paths.sort()

                label_seq_path = os.path.join(self.groundtruth_root, seq_name)
                label_frames = os.listdir(label_seq_path)
                label_frame_paths = [os.path.join(label_seq_path, frame) for frame in label_frames]
                label_frame_paths.sort()

                labels = self.parse_groundtruth(label_frame_paths)

                for i in range(len(image_frames) - (self.num_frames - 1)):
                    sample = image_frame_paths[i:i + self.num_frames]
                    self.samples.append(sample)
                    self.total_len += 1

                    gt = labels[i:i + self.num_frames]
                    self.gts.append(gt)
        else:
            # TODO : test
            pass

    def __getitem__(self, index):
        if (self.phase == 'train') or (self.phase == 'val'):
            sample = self.samples[index]
            sample = [Image.open(img_path) for img_path in sample]

            gt = self.gts[index]
            if self.transform is not None:
                sample, gt = self.transform((sample, gt))

            gt = np.array(gt, dtype=np.float)
            gt = torch.FloatTensor(gt)

            return sample, gt

    def __len__(self):
        return self.total_len

    def parse_groundtruth(self, paths):
        labels = []
        for xml_path in paths:
            label = []
            tree = ET.ElementTree(file=xml_path)
            root = tree.getroot()

            # load frame size
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            for obj in root.iterfind('object'):
                # load labels
                xmax = int(obj.find('bndbox/xmax').text)
                xmin = int(obj.find('bndbox/xmin').text)
                ymax = int(obj.find('bndbox/ymax').text)
                ymin = int(obj.find('bndbox/ymin').text)

                # load class number
                className = obj.find('name').text
                classNo = self.dict[className]

                # lower-left to upper-right
                bbox = [xmin, ymin, xmax, ymax, width, height, classNo]
                label.append(bbox)

            labels.append(label)
        return labels
