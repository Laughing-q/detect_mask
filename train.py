from models import *
from utils.utils import *
from utils.parse_config import *
from utils.datasets import *
from test import evaluate

from terminaltables import AsciiTable

import os
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gradient_accumulations', type=int, default=2, help='number of gradient accums before step')
    parser.add_argument('--model_def', type=str, default='config/yolov3_mask.cfg',
                        help='path to model definition file')
    parser.add_argument('--data_config', type=str, default='config/mask.data', help='path to data config file')
    parser.add_argument('--pretrained_weights', type=str, default='checkpoints/yolov3_ckpt_3.pth',
                        help='if specified starts from checkpoint model')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
    parser.add_argument('--evaluation_interval', type=int, default=1, help='interval evaluations on validation set')
    parser.add_argument('--compute_map', default=False, help='if True computes mAP every tenth batch')
    parser.add_argument('--multiscale_training', default=True, help='allow for multi-scale training')
    opt = parser.parse_args()
    if opt.model_def == 'config/yolov3-tiny_mask.cfg':
        opt.batch_size = 16
    print(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config['train']
    valid_path = data_config['test']
    class_names = load_classes(data_config['names'])
    # print(class_names)

    # 初始化model
    model = Darknet(opt.model_def, opt.img_size).to(device)
    # print(model)
    # model.freeze(cutoff=75)
    # for i in model.children():   # i 为Darknet中的ModuleList()
    #     for j in i.children():      # j为ModuleList()中的Sequential()
    #         for k in j.children():  # k为Sequential()中的Conv2d、BatchNorm2d、LeakyReLU、YOLOLayer等
    #             # print(k.__class__.__name__)
    #             # print(k.__class__.__name__.find('Conv'))
    #             print(k.parameters())
    # 设置参数初始化方式
    model.apply(weights_init_normal)
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith('.pth'):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size, augment=False, multiscale=opt.multiscale_training,
                          normalized_labels=False)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        # collate_fn，是用来处理不同情况下的输入dataset的封装(如何取样本)，一般采用默认即可，除非你自定义的数据读取输出非常少见。
        # 在这里处理给每个batch的每个target编号的问题， 以及多尺度训练
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batch_done = len(dataloader) * epoch + batch_i + 1
            # print(targets)
            imgs, targets = imgs.to(device), targets.to(device)
            # imgs = Variable(imgs.to(device))
            # targets = Variable(targets.to(device), requires_grad=False)
            # print(targets)
            loss, outputs = model(imgs, targets)
            total_loss += loss
            loss_nolized = loss / opt.gradient_accumulations
            loss_nolized.backward()

            if batch_done % opt.gradient_accumulations == 0:
                optimizer.step()
                optimizer.zero_grad()
            if batch_i % 10 == 0 or batch_i == (len(dataloader) - 1):
                print('epoch{}, {}/{}, loss: {}'.format(epoch, batch_i, len(dataloader), loss.float()))
        currect_time = time.time()
        train_time = datetime.timedelta(seconds=currect_time - start_time)
        print('epoch{}, total loss: {}, train time: {}'.format(epoch, total_loss / len(dataloader), train_time))

        if epoch % opt.evaluation_interval == 0:
            print('\n---- Evaluating Model ----')
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            # print(AP, ap_class)
            evaluation_metrics = [
                ('val_precision', precision.mean()),
                ('val_recall', recall.mean()),
                ('val_mAP', AP.mean()),
                ('val_f1', f1.mean()),
            ]
            ap_table = [['Index', 'Class name', 'AP']]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f'---- mAP {AP.mean()}')
        #
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
