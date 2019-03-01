from torch.utils.data import Dataset
import codecs, json
import torch
import networkx as nx
from torch_geometric.data import Data
import os.path as osp


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
        self.datas = [self.subject_to_data(self.G_path, subject, self.scale) for subject in self.subject_ids]
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

    def subject_to_data(self, data_dir, subject_id, scale):
        """
        given a subject's id and scale, return a Data object to be used in torch_geometric
        """
        graph_file = osp.join(data_dir, 'subj{0}.scale{1}.json'.format(subject_id, scale))
        G = nx.node_link_graph(json.load(codecs.open(graph_file, 'r')))

        data = self._networkx_to_data(G)
        # label for patient and non-patient
        data.y = torch.tensor(1) if subject_id[0] == '2' else torch.tensor(0)
        # pre-compute adj to speed things up in DIFFPOOL
        data.adj = self._to_adj(data.edge_index, data.edge_attr, data.num_nodes)

        return data

    @staticmethod
    def _networkx_to_data(G):
        """
        convert a networkx graph to torch geometric data
        """
        # node features
        x = list()
        for i in G.nodes:
            node_feature = [value for key, value in G.nodes[i].items() \
                            if key not in ['StructName', 'FoldInd', 'CurvInd', 'TimeSeries']]
            #         append timeseries to node features
            #         for t in G.nodes[i]['TimeSeries']:
            #             node_feature.append(t)
            x.append(node_feature)

        edge_index = list()
        for edge in G.edges:
            edge_index.append(list(edge))

        edge_attr = list()
        for edge in edge_index:
            edge_feature = list()
            for key in G[edge[0]][edge[1]].keys():
                edge_feature.append(G[edge[0]][edge[1]][key])
            edge_attr.append(edge_feature)

        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def _to_adj(edge_index, edge_attr, num_nodes):
        """
        Return:
            adj: Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]
        """
        # change divice placement to speed up
        adj = torch.zeros(num_nodes, num_nodes, edge_attr.shape[-1])
        i = 0
        for (u, v) in edge_index.transpose(0, 1):
            adj[u][v] = edge_attr[i]
            adj[v][u] = edge_attr[i]
            i = i + 1
        adj = adj.permute(2, 0, 1)
        return adj
