from torch.utils.data import Dataset
import codecs
from utils import subject_to_data


class MmDataset(Dataset):
    def __init__(self,
                 subject_id_path=str,
                 G_path=str,
                 transform=None,
                 scale="60",
                 ):
        """
        Args:
            subject_id_path: path to subject_id file
            transform: pytorch transforms
            scale: ['60', '125', '250'], Lausanne atlas scale
        """
        self.transform = transform
        self.G_path = G_path
        self.scale = scale
        self.subject_ids = [line.strip() for line in codecs.open(subject_id_path, 'r').readlines()]
        print("Loading dataset from disk into memory...")
        self.datas = [subject_to_data(self.G_path, subject, self.scale) for subject in self.subject_ids]
        self.active_datas = self.datas

        for data in self.datas:
            data.x = data.x if self.transform is None else self.transform(data.x)

    @property
    def num_features(self):
        return self[0].num_features

    def __getitem__(self, index):
        return self.active_datas[index]

    def __len__(self):
        return len(self.active_datas)

    def set_active_data(self, index):
        """
        Set active data for K-Folds cross-validation
        Args:
            index: indices for the split
        """
        self.active_datas = [self.datas[i] for i in index]