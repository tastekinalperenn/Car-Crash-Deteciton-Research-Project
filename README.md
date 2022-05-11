# Car Crash Anomaly Detection 

## Introduction

Our project focused on detecting car crashes, which we attempted to do using optical flow, pre-trained convolutional neural networks for extract feature, the You Only Look Once (YOLO) model, recurrent neural networks, and a car crash dataset that included 3000 normal and 1500 crash videos.

The goal of this project was to train a recurrent neural network model to categorize vehicle crash scenarios and observe how it performs with different types of training data based on optical flow, such as magnitude, orientation, magnitude and orientation, and magnitude,orientation with YOLO aspects of the dataset.We will prepare four different datasets and deal with the performance of the same RNN model on these datasets. Thus, we will measure the performance of optical flow types and some features that we extracted from the YOLO object detection algorithm in crash detection.

## Work Done
### Dataset

For this project, we found a dataset of videos of car crashes with 3000 normal videos and 1500 car crash videos. The dataset contain sample videos from day/night and different weather conditions. Information about whether there was an accident for each video and in which frame the accident occurred was shared with us in the dataset.Each videos have 50 frames and 5 second / 10 fps. Our dataset link is here, you can get more detail with visit this link.We will take the videos in this dataset, bring them and convert the format which we work, create four different datasets and develop the RNN model.
https://github.com/Cogito2012/CarCrashDataset

![frame24](https://user-images.githubusercontent.com/59515015/167937146-e0bd7b90-dcab-4ed3-b093-992713844a42.jpg)


### Dataset Preparation

Firstly we extract frames from videos and we have 50 frame images for each videos.While doing this we used dense optical flow method called Farneback Optical Flow. Thus our dataset has been reduces to 24 frames for each videos. Now we have 4500 videos and each videos has 24 optical flow output frames. While doing this we extract optical flow outputs in 3 types which are magnitude based, orientation based  and magnitude*orientation based. 
