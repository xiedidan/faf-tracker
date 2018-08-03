from torch.utils.data import *
import Image
import os

class VidDataset(Dataset):
    def __init__(self, root, packs, phase='train', transform=None, target_transform=None, num_frames=5):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.total_len = 0

        if self.phase == 'train':
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
                for seq_name in os.listdir(pack_path):
                    seq_path = os.path.join(pack_path, seq_name)
                    frames = os.listdir(seq_path)
                    frame_paths = [os.path.join(seq_path, frame) for frame in frames]

                    for i in range(len(frames) - (self.num_frames - 1)):
                        gt = frame_paths[i:i + (self.num_frames - 1)]
                        # TODO : load and parse gt
                        self.gts.append(gt)
        else:
            # TODO : val and test
            pass

    def __getitem__(self, index):
        if self.phase == 'train':
            sample = self.samples[index]
            if self.transform is not None:
                sample = self.transform(sample)
            
            gt = self.gts[index]
            if self.target_transform is not None:
                gt = self.target_transform(gt)

            return sample, gt

    def __len__(self):
        return self.total_len
