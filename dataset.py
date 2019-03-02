from torch.utils.data import Dataset
from torch_geometric.data import Data
import codecs
import torch
from utils import subject_to_data


class Normalize(object):
    """
    Normalize a tensor image with mean and standard deviation.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = [torch.tensor(x) for x in mean]
        self.std = [torch.tensor(x) for x in std]

    def __call__(self, tensor):
        return self.z_score_norm(tensor, self.mean, self.std)

    @staticmethod
    def z_score_norm(tensor, mean, std):
        """
        Normalize a tensor with mean and standard deviation.
        Args:
            tensor (Tensor): Tensor image of size [num_nodes, num_node_features] to be normalized.
            mean (sequence): Sequence of means for each num_node_feature.
            std (sequence): Sequence of standard deviations for each num_node_feature.

        Returns:
            Tensor: Normalized tensor.
        """
        for i, _ in enumerate(mean):
            tensor[:, i] = (tensor[:, i] - mean[i]) / std[i]
        return tensor

    @staticmethod
    def min_max_norm(tensor):
        """
        Normalize a tensor to [0,1]
        """
        for i in range(0, tensor.shape[-1]):
            max_v = torch.max(tensor[:, i])
            min_v = torch.min(tensor[:, i])
            tensor[:, i] = (tensor[:, i] - min_v) / (max_v - min_v)
        return tensor


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
        data = self.active_datas[index]
        return data.x, data.edge_index, data.edge_attr, data.adj, data.y

    def __len__(self):
        return len(self.active_datas)

    def set_active_data(self, index):
        """
        Set active data for K-Folds cross-validation
        Args:
            index: indices for the split
        """
        self.active_datas = [self.datas[i] for i in index]

    @staticmethod
    def collate_fn(batch):  # Deprecated
        """
        Note: real support of minibatch.
        :param batch: list of Data object
        :return:
        """
        batch_data = Data()

        for k, _ in batch[0]:
            batch_data[k] = torch.stack([data[k] for data in batch], dim=0)

        return batch_data
