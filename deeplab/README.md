This repo depends on and is a submodule of [tensorflow models](https://github.com/tensorflow/models).
For document tidy purpose, I didn\'t clone them all.  
It helps you to run the deeplab inference to solve pixel-level semantic segmentation task based on cityspace dataset.

## Download the pretrained model  
### [mobilenet model](http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz)<br>
### [xception model](http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz)

## Download dataset from cityspace, and gather the test png paths into test.list
```bash
find path -name "*.png" > test.list
```

## Run the inference
```python
python3 deeplab_inference.py test.list
```

you will get the result png files under /home/YOUR_UNIX_NAME/share/deeplab_xception/
