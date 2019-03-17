from itertools import repeat

from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import codecs
import torch
from utils import subject_to_data
from data.data_utils import read_mm_data, normalize_node_feature_node_wise, zt_edge_attr
import os.path as osp
from os.path import join
from tqdm import tqdm

from data.data_utils import concat_extra_node_feature, \
    set_missing_node_feature, \
    normalize_node_feature_subject_wise, \
    normalize_node_feature_sample_wise_transform, \
    phrase_subject_list, \
    th_edge_attr, \
    set_edge_attr


class MmDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_set_edge_attr=set_edge_attr,
                 pre_concat=None, pre_set_missing=None, pre_th=None, th=0.0,
                 scale='60', r=3, force=False, batch_size=1):
        self.name = name
        self.pre_concat = pre_concat
        self.pre_transform = pre_transform
        self.pre_set_edge_attr = pre_set_edge_attr
        self.pre_set_missing = pre_set_missing
        self.pre_th = pre_th
        self.th = th
        self.scale = scale
        if scale == '60':
            self.num_nodes = 129
        self.r = r
        self.force = force
        self.batch_size = batch_size
        super(MmDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # collate in dataloader is slooooow
        if self.batch_size > 1:
            self._collate()

    @property
    def raw_file_names(self):
        # mc_filtered_subjects test_subject train_subjects
        return ['train_subjects', 'FEAT.linear/', 'Fs.subjects/', 'LABELS.xlsx', 'Lausanne/']

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
        # subject_list = phrase_subject_list(subject_list)

        print("Reading MM data...")
        data_list = read_mm_data(subject_list=subject_list,
                                 fsl_subjects_dir_path=join(self.raw_dir, self.raw_file_names[1]),
                                 fs_subjects_dir_path=join(self.raw_dir, self.raw_file_names[2]),
                                 atlas_sheet_path=join(self.raw_dir, self.raw_file_names[3]),
                                 atlas_dir_path=join(self.raw_dir, self.raw_file_names[4]),
                                 tmp_dir_path=join(self.raw_dir, 'tmp'),
                                 scale=self.scale,
                                 r=self.r,
                                 force=self.force)
        print("Setting edge_attr")
        data_list = self.pre_set_edge_attr(data_list)

        self.data, self.slices = self.collate(data_list)

        # set missing node feature for subcortical regions
        print("set missing data...")
        self.data.x = self.pre_set_missing(self.data.x) if self.pre_set_missing is not None else self.data.x

        # concat adj to node feature
        if self.pre_concat is not None:
            print("concatenating adj to node feature")
            # this will take a long time...
            # python for-loop is incredibly slow, even with multi-processing
            data_list = self.pre_concat([self._get(i) for i in range(self.__len__())])

        # normalization
        print("Normalizing node attributes")
        self.data, self.slices = self.collate(data_list)
        self.data.x = self.pre_transform(self.data.x, N=len(data_list)) \
            if self.pre_transform is not None else self.data.x

        torch.save((self.data, self.slices), self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.cat_dim(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data.x, data.edge_index, data.edge_attr, data.y, data.adj

    def _get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.cat_dim(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    def add_flag_to_edge_index(self):
        """
        recursively add flag(integer) to edge_index to concatenate graphs to a bigger graph (batch)
        refer: https://rusty1s.github.io/pytorch_scatter/build/html/index.html
        :return:
        """
        dim = self.data.cat_dim('edge_index', self.data.edge_index)
        i = 0
        for idx in range(self.__len__()):
            slices = self.slices['edge_index']
            s = list(repeat(slice(None), self.data.edge_index.dim()))
            s[dim] = slice(slices[idx], slices[idx + 1])
            flag = i * self.num_nodes
            self.data.edge_index[s] += flag
            i += 1
            if i == self.batch_size:
                i = 0

    def _collate(self):
        """
        Collate the graphs in the dataset before passing it to dataloader,
        to make it faster to build a _DataLoaderIter
        :return:
        """
        self.add_flag_to_edge_index()
        keys = self.slices.keys()
        for key in keys:
            self.slices[key] = self.slices[key][::self.batch_size]

    def collate_fn(self, data_list):
        """
        for Pytorch DataLoader
        Duang a batch of graph in to a BIG graph
        :param data_list:
        :return:
        """
        data, slices = self.collate(data_list)
        return data

    def collate_fn_multi_gpu(self, device_count, data_list):
        """
        TODO: Deprecated
        for Pytorch DataLoader
        Usage: partial(collate_fn_multi_gpu, device_count)(data_list)
        :param data_list:
        :param device_count: gpu count used
        :return: list of data
        """
        data_chunks = [data_list[i::device_count] for i in range(device_count)]
        collated_data_list = []
        for data_chunk in data_chunks:
            data, slices = self.collate(data_chunk)
            collated_data_list.append(data)

        return collated_data_list

    def set_active_data(self, index):
        """
        copy the dataset by index
        :param index:
        :return:
        """
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__ = self.__dict__.copy()
        copy.data, copy.slices = self.collate([self._get(i) for i in index])
        return copy

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':
    mmm = MmDataset('data/', 'MM',
                    pre_transform=normalize_node_feature_sample_wise_transform,
                    pre_set_missing=set_missing_node_feature,
                    pre_set_edge_attr=set_edge_attr,
                    pre_concat=concat_extra_node_feature,
                    batch_size=1,
                    r=5,
                    force=False
                    )
    mmm.__getitem__(0)
    print()


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


class MmmDataset(Dataset):
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
        return data.x, data.edge_index, data.edge_attr, data.y, data.adj

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
