# This is a project to check whether people wear mask by yolov3
## prerequisites

* Python 3.6+
* PyTorch 1.3+
* Torchvision 0.4.0+ 
* requirements.txt 

## Datasets
[https://github.com/hamlinzheng](https://github.com/hamlinzheng)

## Training
```Shell
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

```Shell
python train.py --data_config config/mask.data --model_def config/yolov3_mask.cfg --pretrained_weights weights/yolov3_weights
```
## Demo:
```Shell
python detect.py --image_folder data/samples
```

### output
![](https://github.com/Laughing-q/detect_mask/blob/master/output/sample1.png)  
![](https://github.com/Laughing-q/detect_mask/blob/master/output/sample4.png)  
![](https://github.com/Laughing-q/detect_mask/blob/master/output/sample6.png)  

## Webcam Demo:
```Shell
python detect_with_carame.py
```
