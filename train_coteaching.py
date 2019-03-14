import torch.nn.functional as F
import numpy as np
import os.path as osp
from functools import partial

import torch
import torch.distributed as dist
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

from utils import get_model_log_dir
import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

from utils import get_model_log_dir


# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # We don't have a idea weather the dataset is noisy or not
    # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
    # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update), torch.sum(loss_2_update)


def make_rate_schedule(n_epoch, forget_rate, num_gradual, exponent):
    """

    :param n_epoch:
    :param forget_rate:
    :param num_gradual: how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.
    :param exponent: exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.
    :return:
    """
    rate_schedule = np.ones(n_epoch) * forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
    print(rate_schedule)
    return rate_schedule


def train_ct_on_cv(model_cls, dataset, dropout=0.0, lr=1e-3,
                   weight_decay=1e-2, num_epochs=200, n_splits=5,
                   use_gpu=True, multi_gpus=False, distribute=False,
                   comment='', tb_service_loc=None, batch_size=1,
                   num_workers=0, pin_memory=False, cuda_device=None,
                   ddp_port='23456', forget_rate=0.5, num_gradual=60, exponent=1):
    """
    TODO: multi-gpu support
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
        print("creating dataloader tor fold {}".format(fold))
        model_A = model_cls(sample_data, dropout=dropout)
        model_B = model_cls(sample_data, dropout=dropout)

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
            print(model_A)
            writer.add_text('model_summary', model_A.__repr__())
            writer.add_text('training_args', str(saved_args))
        optimizer_A = torch.optim.Adam(model_A.parameters(), lr=lr, betas=(0.9, 0.999),
                                       eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        optimizer_B = torch.optim.Adam(model_B.parameters(), lr=lr, betas=(0.9, 0.999),
                                       eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        if distribute:
            model_A = nn.parallel.DistributedDataParallel(model_A.cuda())
            model_B = nn.parallel.DistributedDataParallel(model_B.cuda())
        else:
            raise Exception("Not implemented")

        rate_schedule = make_rate_schedule(num_epochs, forget_rate=forget_rate,
                                           num_gradual=num_gradual, exponent=exponent)

        for epoch in tqdm_notebook(range(num_epochs), desc='Epoch', leave=False):

            for phase in ['train', 'validation']:

                if phase == 'train':
                    model_A.train()
                    model_B.train()
                    dataloader = train_dataloader
                else:
                    model_A.eval()
                    model_B.eval()
                    dataloader = test_dataloader

                # Logging
                running_corrects_A, running_corrects_B = 0, 0
                running_loss_A, running_loss_B = 0.0, 0.0
                epoch_yhat_0, epoch_yhat_1 = torch.tensor([]), torch.tensor([])

                for x, edge_index, edge_attr, y, adj in tqdm_notebook(dataloader, desc=phase, leave=False):

                    if use_gpu and not multi_gpus:
                        x, edge_index, edge_attr, y, adj = \
                            x.to(device), edge_index.to(device), edge_attr.to(device), y.to(device), adj.to(device)

                    y_hat_A, reg_A = model_A(x, edge_index, edge_attr, y, adj)
                    y_hat_B, reg_B = model_B(x, edge_index, edge_attr, y, adj)
                    y = y.view(-1).cuda() if multi_gpus else y
                    loss_A, loss_B = loss_coteaching(y_hat_A, y_hat_B, y, rate_schedule[epoch])

                    if phase == 'train':
                        optimizer_A.zero_grad()
                        loss_A.backward()
                        optimizer_A.step()
                        optimizer_B.zero_grad()
                        loss_B.backward()
                        optimizer_B.step()

                    loss_A_all = criterion(y_hat_A, y)
                    loss_B_all = criterion(y_hat_B, y)
                    _, predicted_A = torch.max(y_hat_A, 1)
                    _, predicted_B = torch.max(y_hat_A, 1)
                    label = y
                    running_loss_A += loss_A_all.item()
                    running_loss_B += loss_B_all.item()
                    running_corrects_A += (predicted_A == label).sum().item()
                    running_corrects_B += (predicted_B == label).sum().item()

                    epoch_yhat_0 = torch.cat([epoch_yhat_0, y_hat_A[:, 0].detach().view(-1).cpu()])
                    epoch_yhat_1 = torch.cat([epoch_yhat_1, y_hat_A[:, 1].detach().view(-1).cpu()])

                epoch_nll_loss_A = running_loss_A / dataloader.__len__()
                epoch_nll_loss_B = running_loss_B / dataloader.__len__()
                epoch_acc_A = running_corrects_A / dataloader.dataset.__len__() / dataloader.dataset.batch_size
                epoch_acc_B = running_corrects_B / dataloader.dataset.__len__() / dataloader.dataset.batch_size

                # printing statement cause tqdm to buggy in jupyter notebook
                # if epoch % 5 == 0:
                #     print("[Model {0} Epoch {1}]\tLoss: {2:.3f}\tAccuracy: {3:.3f}\t[{4}]".format(
                #         fold, epoch, epoch_total_loss, epoch_acc, phase))

                writer.add_scalars('nll_loss_A',
                                   {'{}_nll_loss'.format(phase): epoch_nll_loss_A},
                                   epoch)
                writer.add_scalars('nll_loss_B',
                                   {'{}_nll_loss'.format(phase): epoch_nll_loss_B},
                                   epoch)
                writer.add_scalars('accuracy_A',
                                   {'{}_accuracy'.format(phase): epoch_acc_A},
                                   epoch)
                writer.add_scalars('accuracy_B',
                                   {'{}_accuracy'.format(phase): epoch_acc_B},
                                   epoch)
                writer.add_histogram('hist/{}_yhat_0'.format(phase),
                                     epoch_yhat_0,
                                     epoch)
                writer.add_histogram('hist/{}_yhat_1'.format(phase),
                                     epoch_yhat_1,
                                     epoch)

    print("Done !")
