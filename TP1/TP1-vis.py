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

import random
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

#%% Optical Flow


#Initialisation des images
Start = 850
End = 1600
Intervalle = 3

n = range(Start,End,Intervalle)

list_image_1 = []
list_image_2 = []
list_GT = []

for i in n:

    NB1 = '{:06}'.format(i)
    NB2 = '{:06}'.format(i+Intervalle)
    
    path1 = os.getcwd()+ '/highway/input/in'+ NB1 +'.jpg'
    path2 = os.getcwd()+ '/highway/input/in'+ NB2 +'.jpg'
    path3 = os.getcwd()+ '/highway/groundtruth/gt'+ NB1 +'.png'
    
    image1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)

    truth = mpimg.imread(path3).astype(np.uint8)
    graytruth = cv2.cvtColor(truth,cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhitetruth) = cv2.threshold(graytruth, 0.9, 255, cv2.THRESH_BINARY)
    blackAndWhitetruth = cv2.bitwise_not(blackAndWhitetruth)
    
    list_image_1.append(image1)
    list_image_2.append(image2)
    list_GT.append(blackAndWhitetruth)


#Apply optical flow
list_bgr = []
list_pred = []
for i in range(len(list_GT)):
    
    image1 = list_image_1[i]
    image2 = list_image_2[i]
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
    
pointage = []
for i in range(len(list_GT)):
    pointage.append(getStats(list_pred[i],list_GT[i]))
    
    
#Plot
image_nb = 135
    
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
plt.imshow(list_image_1[image_nb])

ax3 = plt.subplot(2,2,3)
ax3.set_title("Flot optique")
ax3.axes.get_xaxis().set_visible(False)
ax3.axes.get_yaxis().set_visible(False)
plt.imshow(list_bgr[image_nb])

fig.suptitle('F-score = '+str("%.2f" % pointage[image_nb][-1]))


point_arr = np.array(pointage)
F_score = np.mean(point_arr[:,-1])
print("%.3f" % F_score)

#%% Mask-RCNN

#Initialisation des images
Start = 850
End = 875
Intervalle = 5
n = range(Start,End,Intervalle)
Pointage = np.array([])

for j in n:
    NB1 = '{:06}'.format(j)
        
    path1 = 'highway/input/in'+ NB1 +'.jpg'
    path2 = 'highway/groundtruth/gt'+ NB1 +'.png'
    
    args = {'image' : path1,
            'mask_rcnn' : 'mask-rcnn-coco',
            'visualize' : 0,
            'confidence' : 0.5,
            'threshold' : 0.3}
    
    
    # load the COCO class labels our Mask R-CNN was trained on
    labelsPath = os.path.sep.join([args["mask_rcnn"],
    	"object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")
    
    # load the set of colors that will be used when visualizing a given
    # instance segmentation
    colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
    COLORS = open(colorsPath).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")
    
    
    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = os.path.sep.join([args["mask_rcnn"],
    	"frozen_inference_graph.pb"])
    configPath = os.path.sep.join([args["mask_rcnn"],
    	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
    
    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    
    # load our input image and grab its spatial dimensions
    image = cv2.imread(args["image"])
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
    (thresh, Background) = cv2.threshold(Background, 255, 255, cv2.THRESH_BINARY)
    
    
    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > args["confidence"]:
            # clone our original image so we can draw on it
            clone = image.copy()
            
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
            roi = clone[startY:endY, startX:endX]
            visMask = (mask * 255).astype("uint8")
        
        
        # check to see if are going to visualize how to extract the
        # masked region itself
        
            if args["visualize"] > 0:
                # convert the mask from a boolean to an integer mask with
                # to values: 0 or 255, then apply the mask
                visMask = (mask * 255).astype("uint8")
                instance = cv2.bitwise_and(roi, roi, mask=visMask)
                
                # show the extracted ROI, the mask, along with the
                # segmented instance
                
                cv2.imshow("ROI", roi)
                cv2.imshow("Mask", visMask)
                cv2.imshow("Segmented", instance)
            
            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
            roi = roi[mask]
            
            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = random.choice(COLORS)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            
            # store the blended ROI in the original image
            clone[startY:endY, startX:endX][mask] = blended
            Background[startY:endY, startX:endX]=Background[startY:endY, startX:endX]+[visMask]
            
             

    plt.figure(figsize = (10,6))
    ax1 = plt.subplot(2,3,1)
    ax1.set_title("Image originale")
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    ax2 = plt.subplot(2,3,2)
    ax2.set_title("Ground truth")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    truth = mpimg.imread(path2)
    truth = truth.astype(np.uint8)
    graytruth = cv2.cvtColor(truth,cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhitetruth) = cv2.threshold(graytruth, 0.9, 255, cv2.THRESH_BINARY)
    #blackAndWhitetruth = cv2.bitwise_not(blackAndWhitetruth)
    #plt.imshow(blackAndWhitetruth,cmap='Greys',  interpolation='nearest')
    plt.imshow(cv2.cvtColor(blackAndWhitetruth, cv2.COLOR_BGR2RGB))
    
    ax3 = plt.subplot(2,3,3)
    ax3.set_title("RCNN-Mask")
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    plt.imshow(cv2.cvtColor(Background, cv2.COLOR_BGR2RGB))
    
    
    Union = Background + blackAndWhitetruth
    (thresh, Union) = cv2.threshold(Union, 1, 255, cv2.THRESH_BINARY)
    NBUnion = cv2.countNonZero(Union)
    ax4 = plt.subplot(2,3,4)
    ax4.set_title("Union")
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)
    plt.imshow(cv2.cvtColor(Union, cv2.COLOR_BGR2RGB))
    
    InvertedBackground = cv2.bitwise_not(Background)
    InvertedblackAndWhitetruth = cv2.bitwise_not(blackAndWhitetruth)
    Intersection = InvertedBackground+InvertedblackAndWhitetruth 
    (thresh, Intersection) = cv2.threshold(Intersection, 1, 255, cv2.THRESH_BINARY)
    InvertedIntersection = cv2.bitwise_not(Intersection)
    NBIntersection = cv2.countNonZero(InvertedIntersection)
    ax5 = plt.subplot(2,3,5)
    ax5.set_title("Intersection")
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)
    plt.imshow(cv2.cvtColor(InvertedIntersection, cv2.COLOR_BGR2RGB))
    
    Pointage = np.append(Pointage,NBIntersection/NBUnion)
    
MeanPointage = np.mean(Pointage)
print(MeanPointage)


