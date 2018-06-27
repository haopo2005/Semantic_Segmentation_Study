this repo will help you to train the maskrcnn to solve both pixel-level and instance-level semantic segmentation task based on cityspace dataset  

all the trained model and logs can be downloaded from baiduyun  
link:  
https://pan.baidu.com/s/1syHIsfr4WqQZY_lzMxP5ng  
password:  
0b7v

==================================================================================
#how to train Maskrcnn based on cityspace(only Fine annotations)?  
#1.download dataset from https://www.cityscapes-dataset.com/dataset-overview/  
#unzip the package  
#run shell command:  
unzip leftImg8bit_trainvaltest.zip  
unzip gtFine_trainvaltest.zip  

#2.data preprocess, generate new json file  
#put all the json files under gtFine/train into new path json/train  
#put all the png files under leftImg8bit/train into new path png/train  
#do the same thing for the val set.  
#the png file should correspond with the json file.  
#for train set, run shell command:  
cd /home/jst/share/data/gtFine/train  
find . -name "*.json" | xargs  -i mv  {} /home/jst/share/data/json/train/  

cd /home/jst/share/data/leftImg8bit/train   
find . -name "*.png" | xargs  -i mv  {} /home/jst/share/data/png/train/  

#for val set, run shell command:  
cd /home/jst/share/data/gtFine/val  
find . -name "*.json" | xargs  -i mv  {} /home/jst/share/data/json/val/  

cd /home/jst/share/data/leftImg8bit/val  
 find . -name "*.png" | xargs  -i mv  {} /home/jst/share/data/png/val/   

#generate new json for train set and val set, quite big 67MB  
python mydata.py /home/jst/share/data/json/train /home/jst/share/data/png/train  
python mydata.py /home/jst/share/data/json/val /home/jst/share/data/png/val  

#3.download pretrained coco model "mask_rcnn_coco.h5" from the link:     "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5"  

#4.train the network,BACKBONE is resnet 101, 20 CLASS  
python3 cityspace.py train --dataset /home/jst/share/project/tensorflow/Mask_RCNN/data/ --model   /home/jst/share/project/tensorflow/Mask_RCNN/mask_rcnn_coco.h5  

After hours of training, the loss decreases slowly and you will get mask_rcnn_cityspace_XXXX.h5 under your log folder.  

======================================================================  
#how to run the inference to solve the pixel-level task  
#1.get all the test image file,run the shell command  
find /home/jst/share/project/tensorflow/Mask_RCNN/data/leftImg8bit/test/ -name "*.png" > aaa  

#2.run the inference command, and all the result png files will be storaged at "/home/jst/share/val/"  
python3 cityspace_pixel_all.py --modellogs /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/ --model /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/mask_rcnn_cityspace_0005.h5 --image_path /home/jst/share/project/tensorflow/Mask_RCNN/aaa  

#how to run the inference to solve the instance-level task  
python3 cityspace_instances_all.py --modellogs /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/ --model /home/jst/share/project/tensorflow/Mask_RCNN/logs/cityspace20180526T1041/mask_rcnn_cityspace_0005.h5 --image_path /home/jst/share/project/tensorflow/Mask_RCNN/aaa  
