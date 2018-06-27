This repo depends on https://github.com/tensorflow/models  
This repo is a submodule of tensorflow/models. For Document tidy purpose, I dont clone them all.  
this repo will help you to run the deeplab inference to solve  pixel-level semantic segmentation task based on cityspace dataset  
=============================================================  
#1.download the pretrained model  
mobilenet:  
http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz  
xception:  
http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz  

#2.download dataset from cityspace, and gather the test png path into test.list  
find path -name "*.png" > test.list  

#3.run the inference
python3 deeplab_inference.py test.list  

you will get the result png files under /home/jst/share/deeplab_xception/  


