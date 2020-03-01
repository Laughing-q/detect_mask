from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import torch
from torch.utils.data import DataLoader
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, normalized_labels=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # 提取标签
        labels += targets[:, 1].tolist()

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        # print(targets)
        # imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs = imgs.type(torch.FloatTensor).to(device)
        # imgs.requires_grad = False

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres, nms_thres)
        # print(outputs)
        sample_metrics += get_batch_statistics(outputs, targets, iou_thres)
        # print(sample_metrics)
    # Concatenate sample statistics
    # 将各个sample的true_positives, pred_scores, pred_labels拼接在一起
    # true_positives, pred_scores, pred_labels都为shape(one_image_boxes, )的向量
    # 将每个样本的true_positives, pred_scores, pred_labels分别拼接为一个长向量(all_images_boxes, )
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/mask.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_3.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = parse_data_config(opt.data_config)
    train_path = data_config['train']
    valid_path = data_config['test']
    class_names = load_classes(data_config['names'])

    model = Darknet(opt.model_def).to(device)

    if opt.weights_path.endswith('.weight'):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    print('Compute train mAP...')
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=train_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )
    print('Average Precision:')
    for i, c in enumerate(ap_class):
        print(f'+ Class "{c}" ({class_names[c]}) - AP: {AP[i]}')

    print(f'mAP: {AP.mean()}')

    print('Compute test mAP...')
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )
    print('Average Precision:')
    for i, c in enumerate(ap_class):
        print(f'+ Class "{c}" ({class_names[c]}) - AP: {AP[i]}')

    print(f'mAP: {AP.mean()}')
