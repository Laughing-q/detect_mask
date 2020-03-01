from xml.dom.minidom import parse
import xml.dom.minidom
import os


def get_data(x):
    return x.childNodes[0].data


xmls = os.listdir('data/label_nomask')
print(xmls)
# for path in xmls:
#     mask_label = parse(os.path.join('data/label_nomask', path))
#     root = mask_label.documentElement
#
#     objects = root.getElementsByTagName("object")
#     filename = root.getElementsByTagName("filename")[0]
#     print(get_data(filename).split('.'))
#
#     for object in objects:
#         name = object.getElementsByTagName('name')[0]
#         print(get_data(name))
#         box = object.getElementsByTagName('bndbox')[0]
#         xmin = box.getElementsByTagName('xmin')[0]
#         print(get_data(xmin))
#         ymin = box.getElementsByTagName('ymin')[0]
#         print(get_data(ymin))
#         xmax = box.getElementsByTagName('xmax')[0]
#         print(get_data(xmax))
#         ymax = box.getElementsByTagName('ymax')[0]
#         print(get_data(ymax))
#         a = get_data(filename).split('.')[0]
#         b = int(a) + 514
#         c = str(b).zfill(4)
#         with open('data/label/' + c + '.txt', 'a') as f:
#             if name.childNodes[0].data == 'have_mask':
#                 f.write('1 ')
#             else:
#                 f.write('0 ')
#             f.write(f'{get_data(xmin)} {get_data(ymin)} {get_data(xmax)} {get_data(ymax)}')
#             f.write('\n')
