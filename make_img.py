import os
from PIL import Image

imgs = os.listdir('data/image_nomask')
print(imgs)
for img in imgs:
    image = Image.open(os.path.join('data/image_nomask', img))
    # image.show()
    a = img.split('.')[0]
    b = int(a) + 514
    c = str(b).zfill(4)
    image.save('data/image_mask/' + c + '.jpg')
