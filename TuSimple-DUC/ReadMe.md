This repo helps you run a inference model to solve  pixel-level semantic segmentation task based on cityspace dataset.

The mxnet version used is **1.3.0**
```bash
sudo pip install mxnet-cu90 --pre
```


## Download dataset from cityspace, and gather all the test png paths into test.list
```bash
find path -name "*.png" > test.list
```

## Download the pretrained model from [google drive](https://drive.google.com/drive/folders/0B72xLTlRb0SoREhISlhibFZTRmM)
Download everything about cityspace, and put them under *models* folder  
 
  
## Run the inference script 
```python
python predict_full_image.py
```

Then you will get the predicted mask under the *result* folder  
