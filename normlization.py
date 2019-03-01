import torch


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
