# a = r'D:\projects\PyTorch-YOLOv3-master\output'
# print(a)
# b = a.split('projects')
# print(b)
import torch
#
# a = torch.arange(13).repeat(13, 1)
# print(a.shape)
# print(a)
# b = torch.arange(13).repeat(13, 1).t()
# print(b.shape)
# print(b)
c = torch.FloatTensor([(1.9, 2), (2, 3), (3, 4)])
print(c[:, 0:1])
print(c.contiguous())
print(c[..., 0:2].shape)
print(c)
print(c.long())
#
# a = [1, 2, 3, 4, 5]
# print(a[:len(a)])
#
# a = {3: 'asda', 4: 'asda'}
# print(a[3])
# from PIL import Image
# import torchvision.transforms as transforms
# import numpy as np
# img = transforms.ToTensor()(Image.open('data/samples/dog.jpg'))
# print(img.shape)
# img1 = np.array(Image.open('data/samples/dog.jpg'))
# print(img1.shape)

import torch

a = torch.rand(5, 5)
print(a)
b = a > 0.5
print(b)
print(a[~b])
a = torch.rand(3, 5, 5)
b = torch.flip(a, [-1])
print(a)
print(b)
print(a == b)

# a = [[1, 1], [2, 2], [3, 3]]
# print(list(zip(*a)))
# print(*a)

import numpy as np

# a = np.arange(6).reshape((2, 3))
# print(np.cumsum(a))
# a = np.array([[1], [2], [3]])
# b = np.array([[1], [3], [2]])
# print(np.where(a == b))

# import torch
# a = torch.rand(1, 2, 3, 4)
# print(a[torch.ByteTensor(1, 2, 3, 4).fill_(1)])
