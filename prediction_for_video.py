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
import cv2
import custom

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from video_mask import display_instances_mask
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = custom.CustomConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset")

# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
config.display()
 
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
 
# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"
 
# Load validation dataset
dataset = custom.CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
 
# Must call before using the dataset
dataset.prepare()
class_names = dataset.class_names 

with tf.device(DEVICE):
    model = modellib.MaskRCNN(
                              mode="inference", 
                              model_dir=MODEL_DIR,config=config
                              )
custom_weight_path =os.path.join(MODEL_DIR, "custom.h5")
model.load_weights(custom_weight_path, by_name=True)
 
cap = cv2.VideoCapture('ac_remote.mp4') 
# Check if camera opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 

# Read until video is completed 
while(cap.isOpened()): 
      
  # Capture frame-by-frame 
  ret, frame = cap.read() 
  if ret == True: 
    results = model.detect([frame], verbose=1)
    r = results[0]
    masked_image = display_instances_mask(frame, r['rois'],
                                    r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    cv2.imshow('masked image', masked_image)
#    # Press Q on keyboard to  exit 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break
   
  # Break the loop 
  else:  
    break
   
# When everything done, release  
# the video capture object 
cap.release() 
   
# Closes all the frames 
cv2.destroyAllWindows() 
                                                              