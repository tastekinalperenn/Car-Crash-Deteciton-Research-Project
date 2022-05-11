# Car Crash Anomaly Detection 

## Introduction

Our project focused on detecting car crashes, which we attempted to do using optical flow, pre-trained convolutional neural networks, the You Only Look Once (YOLO) model, recurrent neural networks, and a car crash dataset that included 3000 normal and 1500 crash videos.

The goal of this project was to train a recurrent neural network model to categorize vehicle crash scenarios and observe how it performs with different types of training data based on optical flow, such as magnitude, orientation, magnitude and orientation, and magnitude and orientation with YOLO aspects of the dataset.

## Work Done
### Dataset

For this project, we found a dataset of videos of car crashes with 3000 normal videos and 1500 car crash videos. Each videos have 50 frames and 10 fps. Our dataset link is here, you can get more detail with visit this link.
https://github.com/Cogito2012/CarCrashDataset

### Dataset Preparation

Firstly we extract frames from videos and we have 50 frame images for each videos. We used dense optical flow method called Farneback Optical Flow. Thus our dataset has been reduces to 24 frames for each videos. 
