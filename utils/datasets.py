import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from utils.augmentations import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 四维Tensor：传入四元素tuple(pad_l, pad_r, pad_t, pad_b)，
    # 指的是（左填充，右填充，上填充，下填充），其数值代表填充次数
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding, ’constant‘, ‘reflect’ or ‘replicate’三种模式，指的是常量，反射，复制三种模式
    # value：填充的数值，在"contant"模式下默认填充0，mode = "reflect" or "replicate"时没有value参数
    img = F.pad(img, pad, 'constant', value=pad_value)
    return img, pad


def resize(img, img_size):
    # torch.nn.functional.interpolate函数
    # input是一个四维向量，包括batch*depth*h*w， 所以这里使用torch.unsqueeze()在第0维度添加维数1、
    # 上下采样完了之后 再压缩 ，去除维数为1的维度
    image = F.interpolate(img.unsqueeze(0), size=img_size, mode='nearest').squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        # glob通过正则表达式获取folder_path下的图片
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path))
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


# print(ImageFolder('../data/samples/').files)

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        super(ListDataset, self).__init__()
        with open(list_path, 'r') as fp:
            self.img_files = fp.readlines()
        self.label_files = [
            path.replace('images', 'labels').replace('.png', '.txt').replace('.jpeg', '.txt').replace('.jpg', '.txt')
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        # image
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # 处理通道数小于3的图片
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape  # PIL.Image读取图片是(h, w)
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # pad图片为正方形
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        # print(img.shape)
        # padded_h, padded_w = padded_h + 2, padded_w + 2

        # label
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            # np.loadtxt读取txt文件为数组
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            # Extract coordinates for unpadded + unscaled image
            # 获得框左上角和右下角的坐标（未pad、未resize的）
            # x1 = w_factor * (x - w / 2)
            # y1 = h_factor * (y - h / 2)
            # x2 = w_factor * (x + w / 2)
            # y2 = h_factor * (y + h / 2)
            x1 = w_factor * x1
            y1 = h_factor * y1
            x2 = w_factor * x2
            y2 = h_factor * y2
            # print(boxes)
            # 为padding做调整
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # returns (x, y, w, h)
            # 得到经过padding的和归一化的(x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] = (x2 - x1) * w_factor / padded_w
            boxes[:, 4] = (y2 - y1) * h_factor / padded_h
            # print(boxes)
            targets = torch.zeros((boxes.shape[0], 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        # 这里的targets是一个len为batch_size的列表，每个元素是每张图片的一个或者多个框targets
        # zip(*batch)是将每个batch中的img_path, img, targets分别提取出来再打包
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # 给每张图片的框编号，同一张图片上的框编号一样
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)  # 将len为batch_size的列表cat为tensor张量
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])  # 将包含resize的img的列表在新增的第0维度拼接为tensor张量
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

