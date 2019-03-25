import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import networkx as nx
import os.path as osp
import json, codecs
import numpy as np


def get_model_log_dir(comment, model_name):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = osp.join(
        current_time + '_' + socket.gethostname() + '_' + comment + '_' + model_name)
    return log_dir


def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes=None):
    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)
    ones = torch.ones([edge_index.shape[1] - edge_attr.shape[0], edge_attr.shape[1]], dtype=edge_attr.dtype,
                      device=edge_attr.device)
    edge_attr = torch.cat([edge_attr, ones], dim=0)
    assert edge_index.shape[1] == edge_attr.shape[0]
    return edge_index, edge_attr


def multi_adj_to_edge_index(adj):
    """
    Args:
        adj: <class Tensor> Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]
    """
    adj = adj.permute(1, 2, 0)  # [num_nodes, num_nodes, num_edge_features]

    edge_index = list()
    for i in range(0, adj.shape[0]):
        for j in range(i, adj.shape[1]):
            if adj[i][j].sum().item() != 0:
                edge_index.append([i, j])

    edge_attr = list()
    for u, v in edge_index:
        edge_attr.append(adj[u][v])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_attr = torch.stack([e for e in edge_attr], dim=0)

    return edge_index.to(adj.device), edge_attr.to(adj.device)


def adj_to_edge_index(adj):
    """
    Args:
        adj: <class Tensor> Adjacency matrix with shape [num_nodes, num_nodes]
    """
    device = adj.device
    A = coo_matrix(adj.cpu())
    edge_index = torch.tensor(np.stack([A.row, A.col]), dtype=torch.long, device=device)
    edge_attr = torch.tensor(A.data, device=device).unsqueeze(-1)

    return edge_index, edge_attr


def doubly_stochastic_normlization_d(edge_index, edge_attr, num_nodes):
    adj = adj_to_edge_index(edge_index, edge_attr, num_nodes)

    tilde_adj = adj / adj.sum(dim=1)

    for u, (i, j) in enumerate(edge_index.t()):
        E_i_j = 0
        for k in range(0, num_nodes):
            E_i_j += tilde_adj[i][k] * tilde_adj[j][k] / tilde_adj[:, k].sum(dim=0)
        edge_attr[u] = E_i_j

    return edge_attr


def networkx_to_data(G):
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


def subject_to_data(data_dir, subject_id, scale):
    """
    given a subject's id and scale, return a Data object to be used in torch_geometric
    """
    graph_file = osp.join(data_dir, 'subj{0}.scale{1}.json'.format(subject_id, scale))
    G = nx.node_link_graph(json.load(codecs.open(graph_file, 'r')))

    data = networkx_to_data(G)
    # label for patient and non-patient
    data.y = torch.tensor(1) if subject_id[0] == '2' else torch.tensor(0)
    # pre-compute adj to speed things up in DIFFPOOL
    data.adj = edge_to_adj(data.edge_index, data.edge_attr, data.num_nodes)

    return data
