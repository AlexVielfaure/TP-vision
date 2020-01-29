#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:34:44 2020

@author: alexandre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import tensorflow
import cv2

import matplotlib.image as mpimg
import os

import time

#%% Functions

from sklearn.metrics import confusion_matrix

def getStats(im_seg, GT):
    """Return the usual stats for a confusion matrix."""
    
    im_seg = im_seg.ravel()
    GT = GT.ravel()
    
    TP, FP, FN, TN = confusion_matrix(im_seg,GT).ravel()
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    fmeasure = 2.0 * (recall * precision) / (recall + precision)

    return [recall, precision, fmeasure]




#%% Load images
    
#Initialisation des images
Start = 800
End = 805
Intervalle = 3


list_image_1 = []
list_image_2 = []
list_GT = []


for i in range(Start,End):

    NB1 = '{:06}'.format(i)
    NB2 = '{:06}'.format(i+Intervalle)
    
    path1 = os.getcwd()+ '/highway/input/in'+ NB1 +'.jpg'
    path2 = os.getcwd()+ '/highway/input/in'+ NB2 +'.jpg'
    path3 = os.getcwd()+ '/highway/groundtruth/gt'+ NB1 +'.png'
    
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    truth = mpimg.imread(path3).astype(np.uint8)
    graytruth = cv2.cvtColor(truth,cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhitetruth) = cv2.threshold(graytruth, 0.9, 255, cv2.THRESH_BINARY)
    blackAndWhitetruth = cv2.bitwise_not(blackAndWhitetruth)
    
    list_image_1.append(image1)
    list_image_2.append(image2)
    list_GT.append(blackAndWhitetruth)


#%% Optical Flow

#Apply optical flow
list_bgr = []
list_pred = []

for i in range(len(list_GT)):
    
    image1 = cv2.cvtColor(list_image_1[i], cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(list_image_2[i], cv2.COLOR_BGR2GRAY)
    truth = list_GT[i]
    
    # Pour tous les pixels de l'image
    flot = cv2.calcOpticalFlowFarneback(image1,image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    

    hsv = np.zeros((len(image1),len(image1[0]),3))
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flot[...,0], flot[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)

    grayImage = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    (thresh, BWimage) = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)
    
    list_bgr.append(bgr)    
    list_pred.append(cv2.bitwise_not(BWimage))
    
pointage_OF = []
for i in range(len(list_GT)):
    pointage_OF.append(getStats(list_pred[i],list_GT[i]))
    
    
#Plot
image_nb = 1
    
fig = plt.figure(figsize = (10,10))
ax4 = plt.subplot(2,2,4)
ax4.set_title("Flot optique N&B")
ax4.axes.get_xaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
plt.imshow(list_pred[image_nb],cmap='Greys',  interpolation='nearest')

ax2 = plt.subplot(2,2,2)
ax2.set_title("Ground truth")
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
plt.imshow(list_GT[image_nb],cmap='Greys',  interpolation='nearest')


ax1 = plt.subplot(2,2,1)
ax1.set_title("Video originale")
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
#Original = mpimg.imread(path1)
plt.imshow(cv2.cvtColor(list_image_1[image_nb], cv2.COLOR_BGR2RGB))

ax3 = plt.subplot(2,2,3)
ax3.set_title("Flot optique")
ax3.axes.get_xaxis().set_visible(False)
ax3.axes.get_yaxis().set_visible(False)
plt.imshow(list_bgr[image_nb])

fig.suptitle('F-score = '+str("%.2f" % pointage_OF[image_nb][-1]))

point_arr_OF = np.array(pointage_OF)
Score_OF = np.mean(point_arr_OF,axis=0)
print("Recall : %.3f" % Score_OF[0])
print("Precision : %.3f" % Score_OF[1])
print("F-score : %.3f" % Score_OF[-1])

#%% Mask-RCNN

list_pred_mask = []

args = {'mask_rcnn' : 'mask-rcnn-coco',
        'confidence' : 0.5,
        'threshold' : 0.3}

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# load the set of colors that will be used when visualizing a given instance segmentation
colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


for j in range(len(list_GT)):
        
    # load our input image and grab its spatial dimensions
    image = list_image_1[j]
    
    (H, W) = image.shape[:2]
    
    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()
    
    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
    print("[INFO] boxes shape: {}".format(boxes.shape))
    print("[INFO] masks shape: {}".format(masks.shape))
    
    Background = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    (thresh, Background) = cv2.threshold(Background, 0, 255, cv2.THRESH_BINARY) 
    
    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > args["confidence"]:

            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                   interpolation=cv2.INTER_NEAREST)
            mask = (mask > args["threshold"])
            
            # extract the ROI of the image
            visMask = cv2.bitwise_not((mask * 255).astype("uint8"))
        
            Background[startY:endY, startX:endX]= visMask
            #Background=cv2.bitwise_not(Background)
            
    list_pred_mask.append(Background)
        
  
pointage_m = []
for i in range(len(list_GT)):
    pointage_m.append(getStats(list_pred_mask[i],list_GT[i]))
    
    
#Plot
image_nb = 0
    
fig = plt.figure(figsize = (10,10))
ax4 = plt.subplot(2,2,4)
ax4.set_title("Mask-RCNN pred")
ax4.axes.get_xaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
plt.imshow(list_pred_mask[image_nb],cmap='Greys',  interpolation='nearest')

ax2 = plt.subplot(2,2,2)
ax2.set_title("Ground truth")
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
plt.imshow(list_GT[image_nb],cmap='Greys',  interpolation='nearest')

ax1 = plt.subplot(2,2,1)
ax1.set_title("Video originale")
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
plt.imshow(cv2.cvtColor(list_image_1[image_nb], cv2.COLOR_BGR2RGB))


fig.suptitle('F-score = '+str("%.2f" % pointage_m[image_nb][-1]))

point_arr_m = np.array(pointage_m)
Score_m = np.mean(point_arr_m,axis=0)
print("Recall : %.3f" % Score_m[0])
print("Precision : %.3f" % Score_m[1])
print("F-score : %.3f" % Score_m[-1])   
            
             



