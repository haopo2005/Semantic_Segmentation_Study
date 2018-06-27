import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)
import zipfile
import shutil
import json
import pprint
from collections import namedtuple
import skimage
# Root directory of the project
ROOT_DIR = os.path.abspath("/home/jst/share/project/tensorflow/Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
group_class = ['persongroup','ridergroup','cargroup','truckgroup','busgroup','motorcyclegroup','bicyclegroup','traingroup']
valid_group_class = ['person','rider','car','truck','bus','motorcycle','bicycle','train']
name2label = { label.name : label for label in labels }
valid_class_id = [7,8,24,25,26,27,28,32,33,11,12,13,17,20,19,21,22,23,31]
class cityspaceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "cityspace"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 500

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8
    # BACKBONE = "resnet50"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 19  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class cityspaceDataset(utils.Dataset):
    def load_cityspace(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("cityspace", 7, "road")
        self.add_class("cityspace", 8, "sidewalk")
        self.add_class("cityspace", 24, "person")
        self.add_class("cityspace", 25, "rider")
        self.add_class("cityspace", 26, "car")
        self.add_class("cityspace", 27, "truck")
        self.add_class("cityspace", 28, "bus")
        self.add_class("cityspace", 32, "motorcycle")
        self.add_class("cityspace", 33, "bicycle")
        self.add_class("cityspace", 11, "building")
        self.add_class("cityspace", 12, "wall")
        self.add_class("cityspace", 13, "fence")
        self.add_class("cityspace", 17, "pole")
        self.add_class("cityspace", 20, "traffic sign")
        self.add_class("cityspace", 19, "traffic light")
        self.add_class("cityspace", 21, "vegetation")
        self.add_class("cityspace", 22, "terrain")
        self.add_class("cityspace", 23, "sky")
        self.add_class("cityspace", 31, "train")
        
        # Train or validation dataset?

        dataset_dir = os.path.join(dataset_dir, subset)
        temp = json.load(open(dataset_dir))
        annotations = temp['cityspace']

        for key,a in annotations.items():
           width = a['w']
           height = a['h']   
           instances = a['instances']
           image_path = a['path']
           temp = image_path.split('.')[0].split('/')
           filename = temp[len(temp)-1]
           self.add_image(
                "cityspace",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                instances=instances)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cityspace image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cityspace":
            return super(cityspaceDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["instances"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        count = 0
        for index in annotations:
            mlabel = annotations[index]['label']
            if 'group' in mlabel and mlabel != 'polegroup':
                mlabel = valid_group_class[group_class.index(mlabel)]
            id = name2label[mlabel].id
            if id not in valid_class_id:
                continue
            class_id = self.map_source_class_id("cityspace.{}".format(id))
            if class_id:
                count = count + 1
        instance_masks = np.zeros([image_info["height"], image_info["width"], count], dtype=np.uint8)
        k = 0
        for index in annotations:
            mlabel = annotations[index]['label']
            if 'group' in mlabel and mlabel != 'polegroup':
                mlabel = valid_group_class[group_class.index(mlabel)]
            id   = name2label[mlabel].id
            if id not in valid_class_id:
                continue
            class_id = self.map_source_class_id("cityspace.{}".format(id))
            if class_id:
                rr, cc = skimage.draw.polygon(annotations[index]['shape_attributes']['all_points_y'], annotations[index]['shape_attributes']['all_points_x'])
                instance_masks[rr, cc, k] = 1
                class_ids.append(class_id)
                k = k + 1
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(cityspaceDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the cityspace Website."""
        info = self.image_info[image_id]
        if info["source"] == "cityspace":
            return info["path"]
        else:
            super(cityspaceDataset, self).image_reference(image_id)

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on  cityspace.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' on  cityspace")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/cityspace/",
                        help='Directory of the cityspace dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = cityspaceConfig()
    else:
        class InferenceConfig(cityspaceConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    #model.load_weights(model_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = cityspaceDataset()
        dataset_train.load_cityspace(args.dataset, "train.json")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = cityspaceDataset()
        dataset_val.load_cityspace(args.dataset, "val.json")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
       
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))