This repo uses model voting schema to solve the pixel-level semantic segmentation task based on cityspace dataset.

## Fetch data
- you can download the predict mask images of validation set [here](https://pan.baidu.com/s/1Yq293cniEPcK0XJ5quKXuQ). password:baax  

- unzip these three package, and gather their path
```bash
find /path/deeplab -name "*.png" > deeplab.list  
find /path/erfnet -name "*.png" > erfnet.list  
find /path/tusimple -name "*.png" > tusimple.list  
```

```python
python merge.py deeplab.list tuseng.list erfnet.list
```

## Estimate model
- we can use the cityspace tools to estimate our ensemble model.

## Results
- The best iou of our single model is 79.2  
- We have gained 0.4 improvement by using ensemble model(79.6)  
