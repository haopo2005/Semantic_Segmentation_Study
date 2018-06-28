this repo uses model voting schema to solve the pixel-level semantic segmentation task based on cityspace dataset  
you can download the predict mask images of validation set from baiduyun  
link:  
https://pan.baidu.com/s/1Yq293cniEPcK0XJ5quKXuQ  
password:  
baax  

unzip these three package, and gather their path  
find /path/deeplab -name "*.png" > deeplab.list  
find /path/erfnet -name "*.png" > erfnet.list  
find /path/tusimple -name "*.png" > tusimple.list  

python merge.py deeplab.list tuseng.list erfnet.list

we can use the cityspace tools to estimate our ensemble model  
The best iou of our single model is 79.2  
we have gained 0.4 improvement based on our ensemble model(79.6)  