this repo will help you to run its inference model to solve  pixel-level semantic segmentation task based on cityspace dataset  
the mxnet version is 1.3.0  
sudo pip install mxnet-cu90 --pre  


#1.download dataset from cityspace, and gather all the test png path into test.list  
find path -name "*.png" > test.list  

#2.download the pretrained model from the google drive,download everything about cityspace,and put them under models folder  
https://drive.google.com/drive/folders/0B72xLTlRb0SoREhISlhibFZTRmM  
  
#3.run the inference script  
python predict_full_image.py  

Then you will get the predicted mask under the result folder  

