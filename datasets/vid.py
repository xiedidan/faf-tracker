from torch.utils.data import Dataset

class VidDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass