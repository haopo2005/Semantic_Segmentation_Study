This repo helps you train the maskrcnn to solve both pixel-level and instance-level semantic segmentation task based on cityspace dataset.

All trained models and logs can be found [here](https://pan.baidu.com/s/1syHIsfr4WqQZY_lzMxP5ng), password:0b7v 

# How to train Maskrcnn based on cityspace(only Fine annotations)?
## Fetch data
- Download dataset from [cityscapes](https://www.cityscapes-dataset.com/dataset-overview/)
- unzip the package  
```bash
unzip leftImg8bit_trainvaltest.zip  
unzip gtFine_trainvaltest.zip 
```

## Preprocess data 
- put all the json files under gtFine/train into new path json/train  
- put all the png files under leftImg8bit/train into new path png/train  
- do the same for the val set. The png files should be in the same order as the json files.  
- for train set
```bash
cd /home/jst/share/data/gtFine/train  
find . -name "*.json" | xargs  -i mv  {} /home/jst/share/data/json/train/  

cd /home/jst/share/data/leftImg8bit/train   
find . -name "*.png" | xargs  -i mv  {} /home/jst/share/data/png/train/
```

- for val set
```bash
cd /home/jst/share/data/gtFine/val  
find . -name "*.json" | xargs  -i mv  {} /home/jst/share/data/json/val/  

cd /home/jst/share/data/leftImg8bit/val  
find . -name "*.png" | xargs  -i mv  {} /home/jst/share/data/png/val/
```

- generate new json for train set and val set, quite big ~67MB
```python
python mydata.py /home/jst/share/data/json/train /home/jst/share/data/png/train  
python mydata.py /home/jst/share/data/json/val /home/jst/share/data/png/val
```

## Download pretrained coco model
[mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5)

## Train the network
```python
python3 cityspace.py train --dataset /home/jst/share/project/tensorflow/Mask_RCNN/data/ --model   /home/jst/share/project/tensorflow/Mask_RCNN/mask_rcnn_coco.h5 
```
- BACKBONE is resnet 101
- 20 CLASS  
- After hours of training, the loss decreases slowly and you will get mask_rcnn_cityspace_XXXX.h5 under your log folder.  


# How to run the inference to solve the pixel-level task  
## Get all the test image files
```bash
find /home/jst/share/project/tensorflow/Mask_RCNN/data/leftImg8bit/test/ -name "*.png" > aaa  
```

## Run the inference command
```python
python3 cityspace_pixel_all.py --modellogs /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/ --model /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/mask_rcnn_cityspace_0005.h5 --image_path /home/jst/share/project/tensorflow/Mask_RCNN/aaa  
```
- all the result png files will be storaged at "/home/jst/share/val/"

# How to run the inference to solve the instance-level task
```python
python3 cityspace_instances_all.py --modellogs /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/ --model /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/mask_rcnn_cityspace_0005.h5 --image_path /home/jst/share/project/tensorflow/Mask_RCNN/aaa 
```
