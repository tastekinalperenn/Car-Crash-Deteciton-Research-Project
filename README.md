# Car Crash Anomaly Detection 

## Introduction

Our project focused on detecting car crashes, which we attempted to do using optical flow, pre-trained convolutional neural networks for extract feature, the You Only Look Once (YOLO) model for car detection, recurrent neural networks, and a car crash dataset that included 3000 normal and 1500 crash videos.

The goal of this project was to train a recurrent neural network model to categorize vehicle crash scenarios and observe how it performs with different types of training data based on optical flow variance, such as magnitude, orientation, magnitude and orientation, and  finally magnitude*orientation with YOLO aspects of the dataset.We will prepare four different datasets and deal with the performance of the same RNN model on these datasets. Thus, we will measure the performance of optical flow types and some features that we extracted from the YOLO object detection algorithm in crash detection.

## Work Done
### Dataset

For this project, we found a dataset of videos of car crashes with 3000 normal videos and 1500 car crash videos. The dataset contain sample videos from day/night and different weather conditions. Information about whether there was an accident for each video and in which frame the accident occurred was shared with us in the dataset.Each videos have 50 frames and 5 second / 10 fps. Our dataset link is here, you can get more detail with visit this link. We will take the videos in this dataset, bring them and convert the format which we work, create four different datasets and develop the RNN model.
https://github.com/Cogito2012/CarCrashDataset

![frame24](https://user-images.githubusercontent.com/59515015/167937146-e0bd7b90-dcab-4ed3-b093-992713844a42.jpg)


### Dataset Preparation

Firstly we extract frames from videos and actually we have 50 frame images for each videos.While doing this we used dense optical flow method called Farneback Optical Flow. Since using 50 frame images for each video would extend the file size we will be working with, we skipped one frame and subtracted 24 optical flow frames from each video. (In the beginning our frame number is like this 0,1,2,3,4,....49 and  we used frames numbered 2,4,6,8...) Thus our dataset has been reduces to 24 optical flow frames for each videos. Now we have 4500 videos and each videos has 24 optical flow output frames. While doing this we extract optical flow outputs in 3 types which are magnitude based, orientation based  and magnitude*orientation based. After extracting optical flow operations, we extract feature from oprtical flow frame with using VGG-16 Convolutional Neural Network  fully connected layer and get (1,4096) feature array for each optical flow frame. Here we have 4500 videos and each videos has 24 feature arrays in format (1,4096) which are came VGG-16. We did this operation all of 3 types in optical flows and we get same format all of them.

![vgg-16](https://user-images.githubusercontent.com/59515015/167940559-c709e50e-ee89-43ea-b7a3-aa42b23b1be0.png)

In YOLO part, we work on related 24 frames on original frame images (not optical flow) and detect cars.We calculate 3 different values at this point. First value is number of car count in the video frame images, second value is minumum distance between two nearest cars (if there is no car we initialize same value like -1 for all of video frames) and thirdly we calculate IOU(interseciton over union) with using bounding boxes two nearest cars (if there is no car we did same thing above). And we create our last dataset which is include (1,4096) from magnitude*orientation optical flow features and (3,1) from YOLO and finally we have (1,4099) for each 24 feature array in all of videos. Now we have 4 dataset from 4500 video and each video have 24 feature arrays and these array shapes are here:
<ol>
  <li>Magnitude based Optical Flow Dataset (1,4096)</li>
  <li>Orientation based Optical Flow Dataset (1,4096)</li>
  <li>Magnitude*Orientation based Optical Flow Dataset (1,4096)</li>
  <li>Magnitude*Orientation based Optical Flow and YOLO Dataset (1,4099)</li>
</ol>




![carrrr](https://user-images.githubusercontent.com/59515015/167941923-6b24e348-0997-48cb-95d7-9f3550af9b9b.png)

Since RNN is used to predict time-varying events, we combined our inputs into packages containing 5 consecutive feature arrays instead of one feature array. Thus, while the model was training, the matrices containing these 5 feature arrays belonged to an input data. In this case, let our feature arrays continue as 0,1,2,3,4,5,6. We created an input data by combining the consecutive frames numbered 0,1,2,3,4 with the features containing the relevant information, then we scrolled one by one and got feature arrays 1,2,3,4,5. In the last case, instead of 24 feature arrays in (1,4096)/(1.4099) format for each video, feature arrays in (20,5,4096)/(20,5,4099) format were created. This will each correspond to a train data in our packet train dataset. Finally we concatenate all of feature package and create our dataset.Although we use Google Colab,our YOLO dataset is too big to RAM and we cannot use all of dataset and we randomly select sample and get smaller dataset.Our datasets' final format are here:
<ol>
  <li>Magnitude based Optical Flow Dataset (90000,5,4096)</li>
  <li>Orientation based Optical Flow Dataset (90000,5,4096)</li>
  <li>Magnitude*Orientation based Optical Flow Dataset (90000,5,4096))</li>
  <li>Magnitude*Orientation based Optical Flow and YOLO Dataset (70000,5,4099)</li>
</ol>


