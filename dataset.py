from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import codecs
import torch
from utils import subject_to_data
from data.data_utils import read_mm_data
import os.path as osp
from os.path import join


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


class MmmDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, scale='60', r=3):
        self.name = name
        self.pre_filter = pre_filter
        self.scale = scale
        self.r = r
        super(MmmDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['mc_filtered_subjects', 'FEAT.linear/', 'Fs.subjects/', 'LABELS.xlsx', 'Lausanne/']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        return

    def process(self):
        """
        process raw data, and save
        :return:
        """
        subject_list = [line.strip() for line in
                        codecs.open(osp.join(self.raw_dir, self.raw_file_names[0]), 'r').readlines()]
        data_list = read_mm_data(subject_list=subject_list,
                                 fsl_subjects_dir_path=join(self.raw_dir, self.raw_file_names[1]),
                                 fs_subjects_dir_path=join(self.raw_dir, self.raw_file_names[2]),
                                 atlas_sheet_path=join(self.raw_dir, self.raw_file_names[3]),
                                 atlas_dir_path=join(self.raw_dir, self.raw_file_names[4]),
                                 tmp_dir_path=join(self.raw_dir, 'tmp'),
                                 scale=self.scale,
                                 r=self.r)
        if data_list[0].y.dim() == 0:  # TODO: remove
            for data in data_list:
                data.y = data.y.unsqueeze(0)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':
    mmm = MmmDataset('data/', 'MM')
    mmm.__getitem__(0)
    print()
