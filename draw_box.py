import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

imgs_path = os.listdir('data/images')
labels_path = os.listdir('data/labels')
random.shuffle(imgs_path)
random.shuffle(labels_path)
# print(imgs_path)
# print(labels_path)

# for img_path, label_path in zip(imgs_path, labels_path):
#     img = Image.open(os.path.join('data/images', img_path))
#     label = np.loadtxt(os.path.join('data/labels', label_path))
#     # print(len(label.shape))
#     label = label if len(label.shape) == 2 else np.expand_dims(label, 0)
#     draw = ImageDraw.Draw(img)
#     for box in label:
#         draw.rectangle((box[1], box[2], box[3], box[4]), fill=None, outline=(255, 0, 0), width=2)
#     plt.imshow(img)
#     plt.pause(1)
font = ImageFont.truetype("font/simsun.ttc", 24)
for img_path in imgs_path:
    img = Image.open(os.path.join('data/images', img_path))
    label_path = img_path.replace('jpg', 'txt')
    label = np.loadtxt(os.path.join('data/labels', label_path))
    # print(len(label.shape))
    label = label if len(label.shape) == 2 else np.expand_dims(label, 0)
    draw = ImageDraw.Draw(img)
    for box in label:
        if box[0] == 1:
            mask = 'mask'
        elif box[0] == 0:
            mask = 'nomask'
        draw.text((box[1], box[2]), mask, fill=(0, 255, 0), font=font)
        draw.rectangle((box[1], box[2], box[3], box[4]), fill=None, outline=(255, 0, 0), width=4)
    plt.imshow(img)
    plt.pause(1)
