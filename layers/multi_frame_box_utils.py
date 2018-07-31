# -*- coding: utf-8 -*-

import torch

def multi_frame_log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.detach().max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
    