import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from PIL import Image
import sys
import os
import pickle
import xml.etree.ElementTree as ET
import multiprocessing as mp
from multiprocessing.dummy import Pool
import numpy as np

from tqdm import tqdm

dict_file = './words.txt'

# helper
def file_exists(path):
    try:
        with open(path) as f:
            return True
    except IOError:
        return False

def load_wordnet_dict(path):
    with open(path) as file:
        wordnet_dict = {}

        for line in file.readlines():
            line.strip("\n")
            arr = line.split("\t")
            wordnet_dict[arr[0]] = arr[1]

        return wordnet_dict

# class mapping should be saved since it maybe unstable
# we have a background class by default
def create_class_mapping(val_annotation_path='./ILSVRC/Annotations/VID/val/'):
    print('\ncreating class mapping from: {}'.format(val_annotation_path))
    seqs = os.listdir(val_annotation_path)
    
    num_classes = 1
    classDict = {'background': 0}

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

# mod from torchvision
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, gts, w, h):
        for t in self.transforms:
            images, gts, w, h = t(images, gts, w, h)
        return images, gts, w, h

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# inplace resize transform
class Resize(object):
    def __init__(self, size):
        self.size = size
        self.image_transform = transforms.Lambda(lambda frames: [transforms.Resize(size)(frame) for frame in frames])

    def __call__(self, images, gts, w, h):
        images = self.image_transform(images)

        for i in range(len(gts)):
            gt = gts[i]
            for j in range(len(gt)):
                bbox = gt[j]

                w_ratio = float(self.size[0]) / w
                h_ratio = float(self.size[1]) / h

                bbox = [
                    bbox[0] * w_ratio,
                    bbox[1] * h_ratio,
                    bbox[2] * w_ratio,
                    bbox[3] * h_ratio,
                    bbox[4]
                ]
                gt[j] = bbox
            gts[i] = gt

        return images, gts, w, h

# inplace convert gt to percentage format
class Percentage(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, gts, w, h):
        for frame_index, frame in enumerate(gts):
            for bbox_index, bbox in enumerate(frame):
                bbox = [
                    bbox[0] / self.size[0],
                    bbox[1] / self.size[1],
                    bbox[2] / self.size[0],
                    bbox[3] / self.size[1],
                    bbox[4]
                ]
                
                gts[frame_index][bbox_index] = bbox

        return images, gts, w, h

# ToTensor wrapper
class ToTensor(object):
    def __init__(self):
        self.image_transform = transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames]))

    def __call__(self, images, gts, w, h):
        # TODO : image channel swapping?
        images = self.image_transform(images)
        gts = [torch.Tensor(frame) for frame in gts]
            
        return images, gts, w, h

class VidDataset(Dataset):
    def __init__(self, root, packs, classDict, num_classes=31, phase='train', transform=None, target_transform=None, num_frames=5):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.total_len = 0

        self.dict = classDict
        self.num_classes = num_classes

        self.image_root = os.path.join(self.root, 'Data/VID/', self.phase)
        self.groundtruth_root = os.path.join(self.root, 'Annotations/VID/', self.phase)

        self.samples = []

        # load dump sample file if it exists
        sample_dump_file = os.path.join(self.root, 'dump/sample.{}.dump').format(self.phase)

        if file_exists(sample_dump_file):
            print('\nloading {} sample dump'.format(self.phase))
            with open(sample_dump_file, 'rb') as file:
                self.samples = pickle.load(file)
                self.total_len = len(self.samples)
        elif self.phase == 'train':
            for pack in packs:
                print('\nlisting pack: {}'.format(pack))
                # get samples and gts
                image_pack_path = os.path.join(self.image_root, pack)
                label_pack_path = os.path.join(self.groundtruth_root, pack)
                seq = os.listdir(image_pack_path)

                for seq_name in tqdm(seq):
                    # image paths
                    image_seq_path = os.path.join(image_pack_path, seq_name)
                    image_frames = os.listdir(image_seq_path)
                    image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                    image_frame_paths.sort()

                    # read w, h from first frame of this seq
                    with Image.open(image_frame_paths[0]) as image:
                        w, h = image.size

                    # label paths
                    label_seq_path = os.path.join(label_pack_path, seq_name)
                    label_frames = os.listdir(label_seq_path)
                    label_frame_paths = [os.path.join(label_seq_path, frame) for frame in label_frames]
                    label_frame_paths.sort()

                    # parse xml lable files
                    labels = [self.parse_groundtruth(label_frame_path, w, h) for label_frame_path in label_frame_paths]

                    # split seq into 3D samples
                    for i in range(len(image_frames) - (self.num_frames - 1)):
                        images = image_frame_paths[i:i + self.num_frames]
                        gts = labels[i:i + self.num_frames]

                        self.samples.append((images, gts, w, h))

                        self.total_len += 1
        elif self.phase == 'val':
            print('\nlisting pack: val')
            # get samples and gts
            seq = os.listdir(self.image_root)

            for seq_name in tqdm(seq):
                # image paths
                image_seq_path = os.path.join(self.image_root, seq_name)
                image_frames = os.listdir(image_seq_path)
                image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                image_frame_paths.sort()

                # read w, h from first frame of this seq
                with Image.open(image_frame_paths[0]) as image:
                    w, h = image.size

                # label paths
                label_seq_path = os.path.join(self.groundtruth_root, seq_name)
                label_frames = os.listdir(label_seq_path)
                label_frame_paths = [os.path.join(label_seq_path, frame) for frame in label_frames]
                label_frame_paths.sort()

                # parse xml lable files
                labels = [self.parse_groundtruth(label_frame_path, w, h) for label_frame_path in label_frame_paths]

                # split seq into 3D samples
                for i in range(len(image_frames) - (self.num_frames - 1)):
                    images = image_frame_paths[i:i + self.num_frames]
                    gts = labels[i:i + self.num_frames]

                    self.samples.append((images, gts, w, h))

                    self.total_len += 1
        else:
            # TODO : test
            pass

    def __getitem__(self, index):
        if (self.phase == 'train') or (self.phase == 'val'):
            images, gts, w, h = self.samples[index]

            images = [Image.open(img_path) for img_path in images]

            if self.transform is not None:
                images, gts, w, h = self.transform(images, gts, w, h)

            return images, gts, w, h

    def __len__(self):
        return self.total_len

    def parse_groundtruth(self, xml_path, width, height):
        label = []
        tree = ET.ElementTree(file=xml_path)
        root = tree.getroot()

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
            bbox = [xmin, ymin, xmax, ymax, classNo]
            label.append(bbox)

        # if no obj exists in this frame,
        # then we have a background label
        if len(label) == 0:
            label.append([0, 0, width, height, 0])

        return label

    def get_sample(self):
        return self.samples
