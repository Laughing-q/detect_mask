import os

train_path = os.listdir('data/train/images')
test_path = os.listdir('data/test/images')

with open('data/train/train.txt', 'a') as f:
    for img_path in train_path:
        f.write(os.path.join('data/train/images', img_path) + '\n')

with open('data/test/test.txt', 'a') as f:
    for img_path in test_path:
        f.write(os.path.join('data/test/images', img_path) + '\n')