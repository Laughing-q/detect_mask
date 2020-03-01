import cv2
from models import *
import torch
from torchvision.transforms import ToTensor
from utils.utils import *
from utils.datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Darknet('config/yolov3_mask.cfg', img_size=416).to(device)
model.load_state_dict(torch.load('checkpoints/yolov3_ckpt_3.pth'))
model.eval()
classes = ['no-mask', 'mask']
c = 0
cap = cv2.VideoCapture(0)
h = int(cap.get(3))
w = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'output/sample.avi', fourcc, 20, (h, w))

while cap.isOpened():
    # ret是布尔值，读取帧数正确返回True，读取文件完毕返回False，frame就是每一帧的图像，是个三维矩阵
    ret, frame = cap.read()
    if ret == True:
        timeF = 1
        # if c % timeF == 0:
        # img = cv2.resize(frame, (416, 416))
        img = ToTensor()(frame)
        img, pad = pad_to_square(img, 0)
        img = resize(img, img_size=416)
        img = torch.unsqueeze(img, dim=0).cuda()
        # print(img.shape)
        with torch.no_grad():
            detections = model(img)
            detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.4)[0]
        if detections is not None:
            detections = rescale_boxes(detections, 416, frame.shape[:2])
            # print(detections.shape)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2)
        cv2.imshow('Video', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
