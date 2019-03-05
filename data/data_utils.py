import itertools
import json
import math
import os.path as osp
import pickle
from functools import partial
from multiprocessing.pool import Pool
from os.path import join
from random import shuffle

import networkx as nx
import numpy as np
import openpyxl
import torch
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from torch_geometric.data import Data

from torch_sparse import coalesce


class NumpyEncoder(json.JSONEncoder):
    """
    to save numpy object in json
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_fs_node_feature(subject, fs_subjects_dir_path, scale, hemi):
    """
    read anatomical properties generate by FreeSurfer to a dict,
    for only one hemisphere
    """
    stats_dict = dict()
    stats_dict.update({'hemi': hemi})

    stats_dict.update({'SubCortical': {'NumVert': 0.0,
                                       'SurfArea': 0.0,
                                       'GrayVol': 0.0,
                                       'ThickAvg': 0.0,
                                       'ThickStd': 0.0,
                                       'MeanCurv': 0.0,
                                       'GausCurv': 0.0,
                                       'FoldInd': 0.0,
                                       'CurvInd': 0.0,
                                       'StructName': 'SubCortical'}
                       })

    stats_file = join(fs_subjects_dir_path, subject, 'stats/{0}h.aparc.myatlas_{1}.stats'.format(hemi, scale))
    with open(stats_file, 'r') as infile:
        for line in infile.readlines():
            if not line.startswith('#'):
                StructName = line.split()[0]
                NumVert, SurfArea, GrayVol, ThickAvg, ThickStd, \
                MeanCurv, GausCurv, FoldInd, CurvInd = [float(i) for i in line.split()[1:]]
                stats_dict.update({StructName: {'NumVert': NumVert,
                                                'SurfArea': SurfArea,
                                                'GrayVol': GrayVol,
                                                'ThickAvg': ThickAvg,
                                                'ThickStd': ThickStd,
                                                'MeanCurv': MeanCurv,
                                                'GausCurv': GausCurv,
                                                'FoldInd': FoldInd,
                                                'CurvInd': CurvInd,
                                                'StructName': StructName}
                                   })
    return stats_dict


def to_node_attr_dict_hemi(subject, fs_subjects_dir_path, atlas_sheet_path, scale, hemi):
    """
    map anatomical properties to node id defined by atlas,
    only one hemisphere
    """
    feature_map_dict = dict()

    stats_dict = read_fs_node_feature(subject, fs_subjects_dir_path, scale, hemi)

    sheet_name = hemi.upper() + '_' + scale
    wb = openpyxl.load_workbook(atlas_sheet_path, read_only=True)

    for row in wb[sheet_name]:
        label_id, label_name = row[0].value, row[1].value
        try:
            feature_map_dict.update({label_id: stats_dict[label_name]})
        except KeyError as e:
            feature_map_dict.update({label_id: stats_dict['SubCortical']})
            # feature_map_dict[label_id]['StructName'] = label_name  #TODO: fix buggy {'StructName': 'amygdala'}

    return feature_map_dict


def to_node_attr_dict(subject, fs_subjects_dir_path, atlas_sheet_path, scale):
    """

    :return: node attribute dict, where node id starts from **1**
    """
    node_attr_dict = dict()
    node_attr_dict.update(to_node_attr_dict_hemi(subject, fs_subjects_dir_path, atlas_sheet_path, scale, 'r'))
    node_attr_dict.update(to_node_attr_dict_hemi(subject, fs_subjects_dir_path, atlas_sheet_path, scale, 'l'))

    return node_attr_dict


def to_node_attr_array(subject, fs_subjects_dir_path, atlas_sheet_path, scale,
                       excluded_attr_list=list(['StructName', 'FoldInd', 'CurvInd'])):
    """
    Read anatomical properties computed by FreeSurfer and form a node feature array
    :param subject:
    :param fs_subjects_dir_path:
    :param atlas_sheet_path:
    :param scale:
    :param excluded_attr_list: (optional) :obj:`list` node features to exclude
    :return: :obj:`ndarray` node attribute array with shape [num_nodes, num_features]
    """
    node_attr_dict = to_node_attr_dict(subject, fs_subjects_dir_path, atlas_sheet_path, scale)

    num_nodes = len(node_attr_dict)
    num_node_features = len([k for k in node_attr_dict[1].keys() if k not in excluded_attr_list])
    node_attr_array = np.zeros((num_nodes, num_node_features))
    for node, one_node_attr_dict in node_attr_dict.items():
        one_node_attr_array = np.asarray(
            [value for key, value in one_node_attr_dict.items() if key not in excluded_attr_list])
        node_attr_array[node - 1] = one_node_attr_array  # node start from 1

    return node_attr_array


def to_time_series(subject, fsl_subjects_dir_path, atlas_dir_path, scale):
    """
    mask preprocessed(FSL) fMRI to time series by nilearn [https://nilearn.github.io/
    :param subject:
    :param fsl_subjects_dir_path:
    :param atlas_dir_path:
    :param scale:
    :return: :obj:`ndarray` time series array (standardized) with shape [num_frames, num_nodes]
    """
    atlas_file_path = osp.join(atlas_dir_path, 'QSDR.scale{}.thick2.MNI152.nii.gz'.format(scale))
    masker = NiftiLabelsMasker(labels_img=atlas_file_path, standardize=True, memory='nilearn_cache')
    subject_fmri_path = osp.join(fsl_subjects_dir_path, subject + '.feat', 'filtered_func_data.nii.gz')
    time_series = masker.fit_transform(subject_fmri_path)
    return time_series


def slice_time_series(time_series, stride=100, length=250):
    """
    slice RESTBLOCK/IPBLOCK, sliding window on the rest
    :param length: sliding window length
    :param stride: sliding window stride
    :param time_series:
    :return:
    """
    # re-sample by combination of blocks
    blocked_slices = []
    for i in range(1, 12):
        start = math.ceil((32 * i - 30) / 3)
        end = math.floor((32 * i) / 3)
        blocked_slices.append(time_series[start:end])
    # re-sample by sliding window
    unknown_rest_blocks = []
    for i in itertools.count():
        start = math.ceil((32 * 11 + stride * i) / 3)
        end = start + math.floor((length + stride * i) / 3)
        if end > len(time_series):
            break
        unknown_rest_blocks.append(time_series[start:end])

    rest_blocks, ip_blocks = blocked_slices[::2], blocked_slices[1::2]
    return rest_blocks, ip_blocks, unknown_rest_blocks


def resample_mm(time_series, r):
    """
    RESTBLOCK 6 choose r, IPBLOCK 5 choose r, unknown BLOCK
    :param time_series:
    :param r: N choose r
    :return:
    """
    rest_blocks, ip_blocks, unknown_rest_blocks = slice_time_series(time_series)
    rest_blocks = [np.concatenate(rest_block, axis=0) for rest_block in itertools.combinations(rest_blocks, r)]
    ip_blocks = [np.concatenate(ip_block, axis=0) for ip_block in itertools.combinations(ip_blocks, r)]
    return list(itertools.product(rest_blocks, ip_blocks, unknown_rest_blocks))


def create_and_save_data(scale, fs_subjects_dir_path, atlas_sheet_path,
                         fsl_subjects_dir_path, atlas_dir_path,
                         correlation_measure, r, tmp_dir_path,
                         subject):
    """
    create torch_geometric :obj:`Data` for one subject with re-sampling
    save the list of :obj:`Data` to tmp_dir_path via pickle
    :param scale:
    :param fs_subjects_dir_path:
    :param atlas_sheet_path:
    :param fsl_subjects_dir_path:
    :param atlas_dir_path:
    :param correlation_measure:
    :param r: N choose r, for re-sampling
    :param tmp_dir_path:
    :param subject:
    :return:
    """
    data_list = []

    node_attr_array = to_node_attr_array(subject, fs_subjects_dir_path, atlas_sheet_path, scale)
    time_series = to_time_series(subject, fsl_subjects_dir_path, atlas_dir_path, scale)

    combo_list = resample_mm(time_series, r)
    shuffle(combo_list)
    for combo in combo_list:  # 3 channels in 1 comb (RESTBLOCK IPBLOCK UNKNOWN)
        corr_list = [correlation_measure.fit_transform([cm])[0] for cm in combo]
        # convert correlation to distance between [0, 1]
        corr_list = [1 - np.sqrt((1 - corr) / 2) for corr in corr_list]
        # make the graph fully-connected for spectrum methods
        for i, corr in enumerate(corr_list):
            corr[corr <= 0] = 1e-6
            assert np.count_nonzero(corr) == corr.shape[0] * corr.shape[1]
            corr_list[i] = corr

        # create torch_geometric Data
        G = nx.from_numpy_array(corr_list[0])
        A = nx.to_scipy_sparse_matrix(G)
        adj = A.tocoo()
        edge_index = np.stack([adj.row, adj.col])
        edge_attr = np.zeros((len(adj.row), len(corr_list)))
        all_corr = np.stack(corr_list, axis=-1)
        for i in range(len(adj.row)):
            edge_attr[i] = all_corr[adj.row[i], adj.col[i]]

        data = Data(x=torch.tensor(node_attr_array, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                    y=torch.tensor([1]) if subject.startswith('2') else torch.tensor([0]))
        data_list.append(data)

    subject_tmp_file_path = osp.join(tmp_dir_path, '{}.pickle'.format(subject))
    print("Saving {} samples on subject {}".format(len(data_list), subject))
    with open(subject_tmp_file_path, 'wb') as pfile:
        pickle.dump(data_list, pfile, protocol=pickle.HIGHEST_PROTOCOL)


def read_mm_data(subject_list, fsl_subjects_dir_path, fs_subjects_dir_path,
                 atlas_sheet_path, atlas_dir_path, scale, tmp_dir_path, r=3,
                 force=False):
    """
    read MM data to an in-memory list of :obj:`Data`
    :param force: overwrite existing pickle file in tmp_dir
    :param tmp_dir_path:
    :param subject_list:
    :param fsl_subjects_dir_path:
    :param fs_subjects_dir_path:
    :param atlas_sheet_path:
    :param atlas_dir_path:
    :param scale:
    :param r: rate for re-sampling at time scale
    :return: :obj:`list` list of :obj:`Data`
    """
    done_subject = []
    if not force:
        for subject in subject_list:
            subject_tmp_file_path = osp.join(tmp_dir_path, '{}.pickle'.format(subject))
            if osp.exists(subject_tmp_file_path):
                done_subject.append(subject)
    job_list = [subject for subject in subject_list if subject not in done_subject]
    print("Processing  Job list: {}".format(len(job_list)), job_list)

    if len(job_list) > 0:
        correlation_measure = ConnectivityMeasure(kind='correlation')
        # multiprocessing
        pool = Pool()
        func = partial(create_and_save_data,
                       scale, fs_subjects_dir_path, atlas_sheet_path,
                       fsl_subjects_dir_path, atlas_dir_path,
                       correlation_measure, r, tmp_dir_path)
        pool.map(func, job_list)
        pool.close()
        pool.join()

    data_list = []
    for subject in subject_list:
        subject_tmp_file_path = osp.join(tmp_dir_path, '{}.pickle'.format(subject))
        with open(subject_tmp_file_path, 'rb') as pfile:
            subject_data_list = pickle.load(pfile)
        data_list = data_list + subject_data_list
    return data_list

def normalize_node_feature(x):
    """
    Normalize node feature for a matrix x
    :param x: :obj:`torch.Tensor` Node feature matrix with shape [num_nodes, num_node_features]
    :return:
    """
    intermediate_tensor = x.abs().sum(dim=1)
    zero_tensor = torch.tensor([0])
    mask = 1 - torch.eq(intermediate_tensor, zero_tensor)  # non-zero vectors
    mean_tensor = torch.mean(x[mask], dim=0)
    std_tensor = torch.std(x[mask], dim=0)

    # set missing value (SubCortical)
    mask = torch.eq(intermediate_tensor, zero_tensor)  # zore vectors
    x[mask] = mean_tensor

    # z-score norm
    x = (x - mean_tensor) / std_tensor

    return x


if __name__ == '__main__':
    nnode_attr_array = to_node_attr_array('3044_1', '/data_59/huze/Fs.subjects', "/data_59/huze/Lausanne/LABELS.xlsx",
                                          '60')
    ttime_series = to_time_series('3044_1', '/data_59/huze/MRS/FEAT.linear', '/data_59/huze/Lausanne', '60')
    ddata_list = read_mm_data(['3044_1'], '/data_59/huze/MRS/FEAT.linear',
                              '/data_59/huze/Fs.subjects',
                              "/data_59/huze/Lausanne/LABELS.xlsx", '/data_59/huze/Lausanne', '60',
                              '/data_59/huze/MDEGCNN/data/raw/tmp', r=3,
                              force=False)
    print(ddata_list)
    print("Done!")