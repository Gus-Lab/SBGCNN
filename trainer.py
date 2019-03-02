import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import os.path as osp
from utils import get_model_log_dir
from torch_geometric.nn import global_sort_pool


def train_cross_validation(model_cls, dataset, dropout=0, lr=1e-3,
                           weight_decay=1e-2, num_epochs=200, n_splits=5,
                           use_gpu=True, multi_gpus=True, comment='',
                           tb_service_loc=None):
    """

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
    :return:
    """
    model_name = model_cls.__name__
    # sample data (torch_geometric Data) to construct model
    sample_data = dataset.datas[0]
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
    dataloader = DataLoader(dataset, shuffle=True, batch_size=device_count)

    print("Training {0} {1} models for cross validation...".format(n_splits, model_name))

    folds, fold = KFold(n_splits=n_splits, shuffle=False), 0
    for train_idx, test_idx in tqdm_notebook(folds.split(dataloader.dataset.datas,
                                                         dataloader.dataset.datas),
                                             desc='models', leave=False):
        fold += 1
        model = model_cls(sample_data, dropout=dropout)
        if fold == 1:
            print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        writer = SummaryWriter(log_dir=osp.join('runs', log_dir_base + str(fold)))
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
                if phase == 'train':
                    model.train()
                    dataloader.dataset.set_active_data(train_idx)
                else:
                    model.eval()
                    dataloader.dataset.set_active_data(test_idx)

                running_loss = 0.0
                running_corrects = 0
                epoch_yhat_0, epoch_yhat_1 = np.array([]), np.array([])

                for i, batch in enumerate(dataloader):
                    x, edge_index, edge_attr, adj, y = batch

                    if use_gpu:
                        x, edge_index, edge_attr, adj, y = \
                            Variable(x.cuda()), Variable(edge_index.cuda()), Variable(edge_attr.cuda()), Variable(
                                adj.cuda()), Variable(y.cuda())

                    y_hat, reg = model(x, edge_index, edge_attr, adj)
                    total_loss = criterion(y_hat, y) + reg

                    if phase == 'train':
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    _, predicted = torch.max(y_hat, 1)
                    # _, label = torch.max(y, 1)
                    label = y
                    running_loss += total_loss.detach()
                    running_corrects += (predicted == label).sum().item()

                    epoch_yhat_0 = np.concatenate([epoch_yhat_0, y_hat[:, 0].view(-1).detach().cpu().numpy()],
                                                  axis=None)
                    epoch_yhat_1 = np.concatenate([epoch_yhat_1, y_hat[:, 1].view(-1).detach().cpu().numpy()],
                                                  axis=None)

                epoch_loss = running_loss / dataloader.__len__()
                epoch_acc = running_corrects / dataloader.dataset.__len__()

                # printing statement cause tqdm to buggy in jupyter notebook
                # if epoch % 5 == 0:
                #     print("[Model {0} Epoch {1}]\tLoss: {2:.3f}\tAccuracy: {3:.3f}\t[{4}]".format(
                #         fold, epoch, epoch_loss, epoch_acc, phase))

                writer.add_scalars('data/{}_loss'.format(phase),
                                   {'Total Loss': epoch_loss},
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
