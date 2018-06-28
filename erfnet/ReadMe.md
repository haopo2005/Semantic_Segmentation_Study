this repo will help you to train the erfnet to solve  pixel-level  semantic segmentation task based on cityspace dataset  
all the dependency follows the requirements.txt  
all the logs and model can be downloaded from baiduyun  
link:  
https://pan.baidu.com/s/1ui5VP0zol5uCVrat5r9Bog  
password:  
du23  

==================================================================  
#how to train erfnet based on cityspace(only Fine annotations)?  
# 1.download dataset from https://www.cityscapes-dataset.com/dataset-overview/  
# 2.install the cityscapes tools, and run the createTrainIdLabelImgs.py to generate trainId png files according to their *_polygons.json under the gtFine folder  
git clone https://github.com/mcordts/cityscapesScripts.git  
export CITYSCAPES_DATASET=/home/jst/share/project/tensorflow/Mask_RCNN-master/data/  
python createTrainIdLabelImgs.py  

attentation: I have changed the trainID-labelID correspondence in cityscapesScripts/cityscapesscripts/helpers/label.py  
their correspondence is defined in the train_py/cfg/cityscapes/data.yaml

# 3. put all the train data under data folder and run the jst_erf_data.py to unify suffix of their filename.  
python jst_erf_data.py /path/to/png/file/  

# 4. rearrange the file as the following folder structure(img->origin png files, lbl->trainId png file)  
data  
	train:  
		lbl  
		img  
	val:   
		lbl  
		img  
	test:  
		lbl  
		img  

# 5.run the training script  
[cnn_train.py -d data.yaml -n net.yaml -t train.yaml -l /tmp/path/to/log/ -p /tmp/path/to/pretrained]  
python3 cnn_train.py -d cfg/cityscapes/data.yaml -n cfg/cityscapes/net_bonnet.yaml -t cfg/cityscapes/train_bonnet.yaml -l logs/  

after hours of training, you will get the trained model under logs/iou folder  
===============================================================================  
#how to run the erfnet inference model?  
# 1. get all the test image file,run the shell command  
find /home/jst/share/project/tensorflow/Mask_RCNN/data/leftImg8bit/test/ -name "*.png" > test.list  

# 2. run the inference script  
[cnn_use.py -l /tmp/path/to/log/ -p /tmp/path/to/pretrained -i /path/to/image]  
python3 cnn_use.py  -l logs -p logs -i /home/jst/share/project/tensorflow/Mask_RCNN-master/data/val.list  

you will get the predicted mask png file under logs folder
