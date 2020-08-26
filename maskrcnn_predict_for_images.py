"""
Libraries used for prediction
"""
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print("ROOT_DIR", ROOT_DIR)
sys.path.append(ROOT_DIR)

# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
# import custom file module
import custom

# Directory to load saved model
MODEL_DIR = os.path.join(ROOT_DIR, "log")
config = custom.CustomConfig()

# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on custom dataset
custom_dataset_path =os.path.join(MODEL_DIR, "custom.h5")
model.load_weights(custom_dataset_path, by_name=True)

dataset = custom.CustomDataset()
CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset")
dataset.load_custom(CUSTOM_DIR, "val")

# Must call before using the dataset
dataset.prepare()
# print("final", dataset.class_names)
# Load a random image from the images folder
class_names = dataset.class_names

# real time testing on an image
import matplotlib.image as mpimg
image = mpimg.imread('download.webp')

# Run object detection
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], 
			    r['class_ids'],class_names, r['scores']
			   )
