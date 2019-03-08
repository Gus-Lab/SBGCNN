import copy
import os.path as osp
from functools import partial
from itertools import cycle

import numpy as np
import torch
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm_notebook
from data.data_utils import normalize_node_feature_sample_wise, concat_adj_to_node_feature
from dataset import MmDataset
from models import Baseline
from utils import get_model_log_dir

from boxx import timeit


def my_iter(data_list):
    for data in data_list:
        yield data


def train_cross_validation(model_cls, dataset, dropout=0.0, lr=1e-3,
                           weight_decay=1e-2, num_epochs=200, n_splits=5,
                           use_gpu=True, multi_gpus=False, distribute=False,
                           comment='', tb_service_loc=None, batch_size=1,
                           num_workers=0, pin_memory=False, cuda_device=None):
    """
    TODO: multi-gpu support
    :param distribute: DDP
    :param cuda_device:
    :param pin_memory: DataLoader args https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    :param num_workers: DataLoader args
    :param model_cls: pytorch Module cls
    :param dataset: pytorch Dataset cls
    :param dropout:
    :param lr:
    :param weight_decay:
    :param num_epochs:
    :param n_splits: number of kFolds
    :param use_gpu: bool
    :param multi_gpus: bool
    :param comment: comment in the logs, to filter runs in tensorboard
    :param tb_service_loc: tensorboard service location
    :param batch_size: Dataset args not DataLoader
    :return:
    """
    saved_args = locals()
    if distribute:  # initialize ddp
        dist.init_process_group('nccl', init_method='tcp://localhost:23456', world_size=1, rank=0)
    model_name = model_cls.__name__
    # sample data (torch_geometric Data) to construct model
    sample_data = dataset._get(0)
    if not cuda_device:
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    else:
        device = cuda_device
    device_count = torch.cuda.device_count() if multi_gpus else 1
    if device_count > 1:
        print("Let's use", device_count, "GPUs!")

    log_dir_base = get_model_log_dir(comment, model_name)
    if tb_service_loc is not None:
        print("TensorBoard available at http://{1}/#scalars&regexInput={0}".format(
            log_dir_base, tb_service_loc))
    else:
        print("Please set up TensorBoard")

    criterion = nn.CrossEntropyLoss()

    print("Training {0} {1} models for cross validation...".format(n_splits, model_name))
    folds, fold = KFold(n_splits=n_splits, shuffle=False), 0
    for train_idx, test_idx in tqdm_notebook(folds.split(list(range(dataset.__len__())),
                                                         list(range(dataset.__len__()))),
                                             desc='models', leave=False):
        fold += 1
        print("creating dataloader tor fold {}".format(fold))
        with timeit(name='create dataloader'):
            model = model_cls(sample_data, dropout=dropout)

            collate_fn = partial(dataset.collate_fn_multi_gpu, device_count) if multi_gpus else dataset.collate_fn
            train_dataloader = DataLoader(dataset.set_active_data(train_idx),
                                          shuffle=True,
                                          batch_size=batch_size * device_count,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)
            test_dataloader = DataLoader(dataset.set_active_data(test_idx),
                                         shuffle=True,
                                         batch_size=batch_size * device_count,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory)

            writer = SummaryWriter(log_dir=osp.join('runs', log_dir_base + str(fold)))
            if fold == 1:
                print(model)
                writer.add_text('data/model_summary', model.__repr__())
                writer.add_text('data/training_args', str(saved_args))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                         eps=1e-08, weight_decay=weight_decay, amsgrad=False)

        # add_graph is buggy, who want to fix it?
        # > this bug is complicated... something related to onnx
        # > https://pytorch.org/docs/stable/onnx.html
        # writer.add_graph(model, input_to_model=dataset.__getitem__(0), verbose=True)

        if distribute:
            model = nn.parallel.DistributedDataParallel(model.cuda())
        elif multi_gpus and use_gpu:
            model = nn.DataParallel(model).to(device)
        elif use_gpu:
            model = model.to(device)

        for epoch in tqdm_notebook(range(1, num_epochs + 1), desc='Epoch', leave=False):

            for phase in ['train', 'validation']:

                if phase == 'train':
                    model.train()
                    # data_iter = train_data_iter
                    dataloader = train_dataloader
                else:
                    model.eval()
                    # data_iter = test_data_iter
                    dataloader = test_dataloader

                # Logging
                running_total_loss = 0.0
                running_corrects = 0
                running_reg_loss = 0.0
                running_nll_loss = 0.0
                epoch_yhat_0, epoch_yhat_1 = torch.tensor([]), torch.tensor([])

                for x, edge_index, edge_attr, y in tqdm_notebook(dataloader, desc=phase, leave=False):

                    if use_gpu and not multi_gpus:
                        x, edge_index, edge_attr, y = \
                            x.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

                    y_hat, reg = model(x, edge_index, edge_attr, y)
                    y = y.view(-1).cuda() if multi_gpus else y
                    loss = criterion(y_hat, y)
                    total_loss = (loss + reg).mean()

                    if phase == 'train':
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    _, predicted = torch.max(y_hat, 1)
                    # _, label = torch.max(y, 1)
                    label = y
                    running_nll_loss += loss.item()
                    running_total_loss += total_loss.item()
                    running_reg_loss += reg.sum().item()
                    running_corrects += (predicted == label).sum().item()

                    epoch_yhat_0 = torch.cat([epoch_yhat_0, y_hat[:, 0].detach().view(-1).cpu()])
                    epoch_yhat_1 = torch.cat([epoch_yhat_1, y_hat[:, 1].detach().view(-1).cpu()])

                epoch_total_loss = running_total_loss / dataloader.__len__()
                epoch_nll_loss = running_nll_loss / dataloader.__len__()
                epoch_acc = running_corrects / dataloader.dataset.__len__() / dataloader.dataset.batch_size
                epoch_reg_loss = running_reg_loss / dataloader.dataset.__len__()

                # printing statement cause tqdm to buggy in jupyter notebook
                # if epoch % 5 == 0:
                #     print("[Model {0} Epoch {1}]\tLoss: {2:.3f}\tAccuracy: {3:.3f}\t[{4}]".format(
                #         fold, epoch, epoch_total_loss, epoch_acc, phase))

                writer.add_scalars('data/{}_loss'.format(phase),
                                   {'Total Loss': epoch_total_loss,
                                    'NLL Loss': epoch_nll_loss,
                                    'Total Reg Loss': epoch_reg_loss} if epoch_reg_loss != 0 else
                                   {'Total Loss': epoch_total_loss},
                                   epoch)
                writer.add_scalars('data/{}_accuracy'.format(phase),
                                   {'Total Accuracy': epoch_acc},
                                   epoch)
                writer.add_histogram('hist/{}_yhat_0'.format(phase),
                                     epoch_yhat_0,
                                     epoch)
                writer.add_histogram('hist/{}_yhat_1'.format(phase),
                                     epoch_yhat_1,
                                     epoch)

    print("Done !")


if __name__ == "__main__":
    dataset = MmDataset('data/', 'MM',
                        pre_transform=normalize_node_feature_sample_wise,
                        pre_concat=concat_adj_to_node_feature,
                        batch_size=1000)
    model = Baseline
    train_cross_validation(model, dataset, comment='test_batch', batch_size=512,
                           num_epochs=500, dropout=0.3, lr=1e-8, weight_decay=1e-2,
                           use_gpu=True, multi_gpus=False, tb_service_loc="",
                           num_workers=28, pin_memory=True)
