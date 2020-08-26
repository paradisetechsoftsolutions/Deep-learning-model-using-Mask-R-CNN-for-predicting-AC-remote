# Deep-learning-model-using-Mask-R-CNN-for-predicting-AC-remote
Mask R-CNN is a simple, flexible, and general framework for object instance segmentation. Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. It is capable of separating different objects in a image or a video. Research paper for Mask R-CNN can be found    ![here](https://arxiv.org/pdf/1703.06870.pdf)

Faster RCNN was not designed for pixel-to-pixel alignment between network inputs and outputs. Faster R-CNN is advanced by learning the attention mechanism with a Region Proposal Network (RPN). Faster R-CNN produce two outputs for each candidate object that is a class label and a bounding-box offset whereas Mask R-CNN outputs object mask along with two. The additional mask output is distinct from the class and box outputs, requiring extraction of much finer spatial layout of an object.

A sample image of masked object from our dataset is shown below:

![prediction](https://user-images.githubusercontent.com/39157936/91271109-4ca23c00-e797-11ea-8f4b-85ed8cc2ece9.png)  

# Network Architecture for this project

* **Convolutional arcitecture** - 

* **Feature extraction** - 

* **Bounding box recognition** - 

* **Mask prediction** - 

# Implementation details of this project

* **App structure** - 

* **Dataset** - 

* **Training** - 

* **Conclusions/Results obtained** - 

# Steps Involved for creating pixel level boundary in an image  

* **Step1:** Download dataset of ac remote. One can use Download All Images (Google Chrome extension). One can add it using the [link](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en)  

* **Step2:** Split the data into training and validation folders. For both train and val folder do annotation using the tool [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html)  

* **Step3:** On Success for the Step2, one will get the json files for both training and validation dataset. Keep them in their respective folder.  

* **Step4:** Now it is the time for the training the model. Trained model will be saved in the log directory which is created in the root directory. Log directory will get automatically created.  

* **Step5:** One can skip all the abbove 4 steps and use the trained weights created by me. Download the trained weights from the [link](https://drive.google.com/file/d/1fj9uxffJ41PQ1Ay0YQfzA75PhGn3D0z7/view?usp=sharing). Place these custom weights for ac remote in the log directory in the root folder.  

* **Step6:** To Do prediction with the image dowbnload any ac remote image from the internet and use the python file maskrcnn_predict_for_images.py.    

* **Step7:** To do prediction on the video file download video file containing ac remote and run the file prediction_for_video.py  

# References
* 
* https://arxiv.org/pdf/1703.06870.pdf
