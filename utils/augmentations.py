import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])   # torch.flip(input, dims) 按照给定维度翻转张量，这里-1为最后一个维度，为左右翻转
    targets[:, 2] = 1 - targets[:, 2]   # 左右翻转, 框的横坐标x翻转
    return images, targets
