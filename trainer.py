import os.path as osp
from functools import partial

import torch
import torch.distributed as dist
from boxx import timeit
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

from data.data_utils import normalize_node_feature_subject_wise, concat_extra_node_feature, \
    normalize_node_feature_node_wise
from dataset import MmDataset
from models import Baseline
from utils import get_model_log_dir


def init_all(model_cls, dataset, dropout=0, lr=1e-3,
             weight_decay=1e-3, use_gpu=True, multi_gpus=True,
             distribute=True, comment='', tb_service_loc=None,
             batch_size=1, num_workers=0, pin_memory=True, cuda_device=None,
             ddp_port='23333', fold_no=None):
    """

    :param fold_no:
    :param model_cls:
    :param dataset:
    :param dropout:
    :param lr:
    :param weight_decay:
    :param use_gpu:
    :param multi_gpus:
    :param distribute:
    :param comment:
    :param tb_service_loc:
    :param batch_size:
    :param num_workers:
    :param pin_memory:
    :param cuda_device:
    :param ddp_port:
    :return:
    """
    saved_args = locals()
    if distribute:  # initialize ddp
        dist.init_process_group('nccl', init_method='tcp://localhost:{}'.format(ddp_port), world_size=1, rank=0)
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
        print("Please set up your TensorBoard")

    criterion = nn.CrossEntropyLoss()


def train_cross_validation(model_cls, dataset, dropout=0.0, lr=1e-3,
                           weight_decay=1e-2, num_epochs=200, n_splits=5,
                           use_gpu=True, multi_gpus=False, distribute=False,
                           comment='', tb_service_loc=None, batch_size=1,
                           num_workers=0, pin_memory=False, cuda_device=None,
                           ddp_port='23456', fold_no=None):
    """
    TODO: multi-gpu support
    :param fold_no:
    :param ddp_port:
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
        dist.init_process_group('nccl', init_method='tcp://localhost:{}'.format(ddp_port), world_size=1, rank=0)
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
    print(dataset.__len__())
    for train_idx, test_idx in tqdm_notebook(folds.split(list(range(dataset.__len__())),
                                                         list(range(dataset.__len__()))),
                                             desc='models', leave=False):
        fold += 1
        if fold_no is not None:
            if fold != fold_no:
                continue
        print("creating dataloader tor fold {}".format(fold))
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
        model_save_dir = osp.join('saved_models', log_dir_base + str(fold))
        if fold == 1:
            print(model)
            writer.add_text('model_summary', model.__repr__())
            writer.add_text('training_args', str(saved_args))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        if distribute:
            model = nn.parallel.DistributedDataParallel(model.cuda())
        elif multi_gpus and use_gpu:
            model = nn.DataParallel(model).to(device)
        elif use_gpu:
            model = model.to(device)

        best_map = 0.0
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

                for x, edge_index, edge_attr, y, adj in tqdm_notebook(dataloader, desc=phase, leave=False):

                    if use_gpu and not multi_gpus:
                        x, edge_index, edge_attr, y, adj = \
                            x.to(device), edge_index.to(device), edge_attr.to(device), y.to(device), adj.to(device)

                    y_hat, reg = model(x, edge_index, edge_attr, y, adj)
                    y = y.view(-1).cuda() if multi_gpus else y
                    loss = criterion(y_hat, y)
                    total_loss = (loss + reg).mean()
                    # total_loss = loss

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

                writer.add_scalars('nll_loss',
                                   {'{}_nll_loss'.format(phase): epoch_nll_loss},
                                   epoch)
                writer.add_scalars('accuracy',
                                   {'{}_accuracy'.format(phase): epoch_acc},
                                   epoch)
                if epoch_reg_loss != 0:
                    writer.add_scalars('reg_loss'.format(phase),
                                       {'{}_reg_loss'.format(phase): epoch_reg_loss},
                                       epoch)
                writer.add_histogram('hist/{}_yhat_0'.format(phase),
                                     epoch_yhat_0,
                                     epoch)
                writer.add_histogram('hist/{}_yhat_1'.format(phase),
                                     epoch_yhat_1,
                                     epoch)

                if phase == 'validation':
                    model_save_path = model_save_dir + '-{}-{}-{:.3f}-{:.3f}'.format(model_name, epoch, epoch_acc,
                                                                                     epoch_nll_loss)
                    if epoch_acc > best_map:
                        best_map = epoch_acc
                        model_save_path = model_save_path + '-best'
                    torch.save(model.state_dict(), model_save_path)

    print("Done !")


if __name__ == "__main__":
    dataset = MmDataset('data/', 'MM',
                        pre_transform=normalize_node_feature_node_wise,
                        pre_concat=concat_extra_node_feature,
                        batch_size=1000)
    model = Baseline
    train_cross_validation(model, dataset, comment='test_batch', batch_size=512,
                           num_epochs=500, dropout=0.3, lr=1e-8, weight_decay=1e-2,
                           use_gpu=True, multi_gpus=False, tb_service_loc="",
                           num_workers=28, pin_memory=True)
