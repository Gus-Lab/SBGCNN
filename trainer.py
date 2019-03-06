import os.path as osp

import numpy as np
import torch
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

from utils import get_model_log_dir

from boxx import timeit

def train_cross_validation(model_cls, dataset, dropout=0, lr=1e-3,
                           weight_decay=1e-2, num_epochs=200, n_splits=5,
                           use_gpu=True, multi_gpus=True, comment='',
                           tb_service_loc=None, batch_size=1,
                           num_workers=0, pin_memory=False):
    """

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
    :param batch_size: DataLoader args, make sure your dataset and model support mini-batch setting
    :return:
    """
    saved_args = locals()
    model_name = model_cls.__name__
    # sample data (torch_geometric Data) to construct model
    sample_data = dataset.__getitem__(0)
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
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
        with timeit(name='create dataloader'):
            fold += 1
            model = model_cls(sample_data, dropout=dropout)

            train_dataloader = DataLoader(dataset.set_active_data(train_idx),
                                          shuffle=True,
                                          collate_fn=dataset.collate_fn,
                                          batch_size=device_count * batch_size,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)
            test_dataloader = DataLoader(dataset.set_active_data(test_idx),
                                         shuffle=True,
                                         collate_fn=dataset.collate_fn,
                                         batch_size=device_count * batch_size,
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

        if multi_gpus and use_gpu:
            model = nn.DataParallel(model).to(device)
        elif use_gpu:
            model = model.to(device)

        for epoch in tqdm_notebook(range(1, num_epochs + 1), desc='Epoch', leave=False):

            for phase in ['train', 'validation']:
                with timeit(name='switch phase'):
                    if phase == 'train':
                        model.train()
                        dataloader = train_dataloader
                    else:
                        model.eval()
                        dataloader = test_dataloader

                    # Logging
                    running_total_loss = 0.0
                    running_corrects = 0
                    running_reg_loss = 0.0
                    running_nll_loss = 0.0
                    epoch_yhat_0, epoch_yhat_1 = np.array([]), np.array([])

                for data in tqdm_notebook(dataloader, desc='DataLoader', leave=False):

                    with timeit(name='input to device'):
                        if use_gpu:
                            for key in data.keys:
                                data[key] = Variable(data[key].cuda())
                                # data[key] = data[key].to(device)

                    y = data.y
                    y_hat, reg = model(data)
                    loss = criterion(y_hat, y)
                    total_loss = (loss + reg).mean()

                    if phase == 'train':
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    _, predicted = torch.max(y_hat, 1)
                    # _, label = torch.max(y, 1)
                    label = y
                    running_nll_loss += loss.detach()
                    running_total_loss += total_loss.detach()
                    running_reg_loss += reg.sum().item()
                    running_corrects += (predicted == label).sum().item()

                    epoch_yhat_0 = np.concatenate([epoch_yhat_0, y_hat[:, 0].view(-1).detach().cpu().numpy()],
                                                  axis=None)
                    epoch_yhat_1 = np.concatenate([epoch_yhat_1, y_hat[:, 1].view(-1).detach().cpu().numpy()],
                                                  axis=None)

                epoch_total_loss = running_total_loss / dataloader.__len__()
                epoch_nll_loss = running_nll_loss / dataloader.__len__()
                epoch_acc = running_corrects / dataloader.dataset.__len__()
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
