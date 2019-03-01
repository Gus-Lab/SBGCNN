import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter


def train_cross_validation(model_cls, dataloader, dropout=0, lr=1e-3,
                           weight_decay=1e-2, num_epochs=200, n_splits=5,
                           use_gpu=True, multi_gpus=True, comment=''):
    """
    Args:
        model: model <class>
        dataloader: pytorch Dataloader
        comment: comment in the logs, to filter runs in tensorboard
    """
    model_name = model_cls.__name__
    # sample data (torch_geometric Data) to construct model
    sample_data = dataloader.dataset.datas[0]
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    criterion = nn.NLLLoss()

    print("Training {0} {1} models...".format(n_splits, model_name))
    print("TensorBoard available at http://localhost:6006/#scalars&regexInput={0}_{1}".format(comment, model_name))

    folds = KFold(n_splits=n_splits, shuffle=False)
    fold = 0
    for train_idx, test_idx in tqdm(folds.split(dataloader.dataset.datas, dataloader.dataset.datas),
                                    desc='models', leave=False):
        fold += 1
        model = model_cls(sample_data, dropout=dropout)
        if fold == 1:
            print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        writer = SummaryWriter(comment='_{0}_{1}{2}'.format(comment, model_name, fold))

        if multi_gpus:
            model = nn.DataParallel(model).to(device)
        elif use_gpu:
            model = model.to(device)

        for epoch in tqdm(range(1, num_epochs + 1), desc='Epoch', leave=False):

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                    dataloader.dataset.set_active_data(train_idx)
                else:
                    model.eval()
                    dataloader.dataset.set_active_data(test_idx)

                running_loss = 0.0
                running_corrects = 0

                for i, batch in enumerate(dataloader):
                    x, edge_index, edge_attr, y = batch[0], batch[1], batch[2], batch[3]
                    if use_gpu:
                        x, edge_index, edge_attr, y = \
                            Variable(x.cuda()), Variable(edge_index.cuda()), Variable(edge_attr.cuda()), Variable(
                                y.cuda())
                    else:
                        x, edge_index, edge_attr, y = \
                            Variable(x), Variable(edge_index), Variable(edge_attr), Variable(y)

                    if model_name == 'EGAT_DIFFPOOL':
                        y_hat, reg1, reg2 = model(x, edge_index, edge_attr)
                        loss = criterion(y_hat, y)
                        total_loss = loss + reg1.sum() / reg1.numel() + reg2.sum() / reg2.numel()
                    else:
                        y_hat = model(x, edge_index, edge_attr)
                        total_loss = criterion(y_hat, y)

                    if phase == 'train':
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                    _, predicted = torch.max(y_hat, 1)
                    # _, label = torch.max(y, 1)
                    label = y
                    running_loss += total_loss.detach()
                    running_corrects += (predicted == label).sum().item()

                epoch_loss = running_loss / dataloader.__len__()
                epoch_acc = running_corrects / dataloader.dataset.__len__()

                writer.add_scalars('data/{}_loss'.format(phase),
                                   {'Total Loss': epoch_loss},
                                   epoch)
                writer.add_scalars('data/{}_accuracy'.format(phase),
                                   {'Total Accuracy': epoch_acc},
                                   epoch)
    print("Done !")
