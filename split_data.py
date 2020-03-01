import random
import os
from PIL import Image
import shutil

imgs_path = os.listdir('data/images')
labels_path = os.listdir('data/labels')
random.shuffle(imgs_path)
print(imgs_path)
# print(labels_path)


# shutil.move('data/images/0000.jpg', 'data/labels')

for path in imgs_path[:300]:
    img_path = os.path.join('data/images', path)
    label_path = os.path.join('data/labels', path.replace('jpg', 'txt'))
    shutil.move(img_path, 'data/test/images')
    shutil.move(label_path, 'data/test/labels')
