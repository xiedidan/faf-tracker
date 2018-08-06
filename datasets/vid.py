from torch.utils.data import *
import Image
import os
import xml.etree.ElementTree as ET

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
    for seq_name in os.listdir(val_annotation_path):
        seq_path = os.path.join(val_annotation_path, seq_name)
        frames = os.listdir(seq_path)
        frame_paths = [os.path.join(seq_path, frame) for frame in frames]

        num_classes = 0
        classDict = {}

        for xml_path in frame_paths:
            tree = ET.ElementTree(file=xml_path)

            for obj in tree.iterfind('annotation/object'):
                # create wnid to classNo mapping
                className = obj.find('name').text
                if classDict[className] is None:
                    classDict[className] = num_classes
                    num_classes += 1
        
        return num_classes, classDict

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

        if (self.phase == 'train') or (self.phase == 'val'):
            self.image_root = os.path.join(root, '/Data/VID/', self.phase)
            self.groundtruth_root = os.path.join(root, '/Annotations/VID/', self.phase)

            self.samples = []
            self.gts = []
            for pack in packs:
                # get image samples
                pack_path = os.path.join(self.image_root, pack)

                for seq_name in os.listdir(pack_path):
                    seq_path = os.path.join(pack_path, seq_name)
                    frames = os.listdir(seq_path)
                    frame_paths = [os.path.join(seq_path, frame) for frame in frames]

                    for i in range(len(frames) - (self.num_frames - 1)):
                        sample = frame_paths[i:i + (self.num_frames - 1)]
                        self.samples.append(sample)
                        self.total_len += 1
                
                # get labels
                pack_path = os.path.join(self.groundtruth_root, pack)

                for seq_name in os.listdir(pack_path):
                    seq_path = os.path.join(pack_path, seq_name)
                    frames = os.listdir(seq_path)
                    frame_paths = [os.path.join(seq_path, frame) for frame in frames]

                    labels = self.parse_groundtruth(frame_paths)

                    # create multi-frame gts
                    for i in range(len(frames) - (self.num_frames - 1)):
                        gt = labels[i:i + (self.num_frames - 1)]
                        self.gts.append(gt)
        else:
            # TODO : test
            pass

    def __getitem__(self, index):
        if self.phase == 'train':
            sample = self.samples[index]
            # read images with PIL
            sample = [Image.open(img_path) for img_path in sample]

            if self.transform is not None:
                sample = self.transform(sample)
            
            gt = self.gts[index]
            if self.target_transform is not None:
                gt = self.target_transform(gt)

            return sample, gt

    def __len__(self):
        return self.total_len

    def parse_groundtruth(self, paths):
        labels = []

        for xml_path in paths:
            label = []
            tree = ET.ElementTree(file=xml_path)

            for obj in tree.iterfind('annotation/object'):
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

            labels.append(label)
        
        return labels
