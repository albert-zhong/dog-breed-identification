---
title: Dog Breed Identification Using Neural Networks
layout: post
date: '2022-06-05 15:12:11 +0800'
categories:
- Projects
tags: python pytorch
---


# Dog Breed Identification

*Created by: Armand Mousavi (amousavi@cs), Vivek Patel (vivekp@cs), and Albert Zhong (azhong@cs)*

*UW student project for CSE455 22sp*

## Video

[![Project Video Presentation](http://img.youtube.com/vi/pWN0kjnHRRs/0.jpg)](https://youtu.be/pWN0kjnHRRs "Video")

## Abstract and Background

FGIC (Fine-Grained Image Classification) is a core problem in modern machine learning research and the discipline of computer vision in particular. The use of image data that is labeled for the purposes of predicting an attribute that is categorical seems clear at first, but presents a huge challenge when considering the amount of possible labels that can be assigned in addition to the distribution of data to both train and test approaches on.

Neural networks (and more specifically, convolutional neural networks) are a key tool used in tackling fine-grained image classification problems. A general neural network architecture for image classification usually involves taking a preprocessed input (common transformations include square-cropping, rotations and zooms on image data to prevent overfitting) and then convolving, activating, and pooling the results to then transform the input into a different shape as to learn higher-order features present in the data. This smaller portion is then often repeated some number of times before one or more fully connected layers with a softmax-esque activation function that yields what can be considered output probabilities for each class of the dependent variable. An example image is presented below.

![An example CNN architecture diagram](https://www.researchgate.net/publication/322848501/figure/fig2/AS:589054651420677@1517452981243/CNN-architecture-used-to-perform-image-classification-Ant-specimen-photograph-by-April.png)
[[Source]](https://www.researchgate.net/figure/CNN-architecture-used-to-perform-image-classification-Ant-specimen-photograph-by-April_fig2_322848501)

## Problem Statement

Our goal is to evaluate multiple common image classification networks that are more general on their ability to perform dog breed identification on the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## Data Source
As mentioned above, we utilized the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). It features 20850 images total across 120 dog breeds. There are roughly 150 images per dog breed in the dataset, a fairly even distribution, with some variation around that number.

## Methodology
In general, the workflow we mentioned in the abstract and background section follows in the approaches we took for our work here. 

1. Gather Dataset
2. Preprocess Training/Validation/Test Datasets
    1. Cropping
    2. Flipping
    3. Rotation
3. Train
4. Evaluate Performance

We compared a few different common general models for image classification:

* ResNet-34
* ResNet-50
* Inception-v3

We leveraged pretrained weights made available by PyTorch, but had to modify the networks to support predictions across 120 labels (the number of different breeds in the dataset). We took the fully-connected layers at the end of each network and removed them, replacing them with layers that have 120 outputs. From there, our training code takes the argmax over the output layer and deems that as the prediction made by the neural network in question.

## Experimental Setup and Results

The full training and testing code we used to comapre the models can be viewed in our Colab notebook [here](https://colab.research.google.com/drive/1n4Donev0PE45W8-coGbfZ-s5n1rdc0x8?usp=sharing).

For each network we trained against the dataset, we generated plots for Training Loss vs. Epoch and Validation Loss vs. Epoch. We utilized a 70-15-15 split for training, validation, and testing. All models were trained for 15 epochs of stochastic gradient descent with a learning rate of 0.01, momentum of 0, and weight decay of 0.0001.

The models we tested performed as follows:

### ResNet-34
![Training Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Training%20Loss%20vs%20Epoch%20-%20Modified%20Resnet-34.png?raw=true)

![Validation Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Validation%20Loss%20vs%20Epoch%20-%20Modified%20Resnet-34.png?raw=true)

The resulting accuracy for the network on the test set was roughly 75.6%.
![](https://github.com/albert-zhong/dog-breed-identification/blob/main/Resnet-34%20Accuracy.png?raw=true)

### ResNet-50
![Training Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Training%20Loss%20vs%20Epoch%20-%20Modified%20Resnet-50.png?raw=true)

![Validation Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Validation%20Loss%20vs%20Epoch%20-%20Modified%20Resnet-50.png?raw=true)

The resulting accuracy for the network on the test set was roughly 79.68%.
![](https://github.com/albert-zhong/dog-breed-identification/blob/main/Resnet-50%20Accuracy.png?raw=true)

### Inception-v3
![Training Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Training%20Loss%20vs%20Epoch%20-%20Modified%20Inception-v3.png?raw=true)

![Validation Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Validation%20Loss%20vs%20Epoch%20-%20Modified%20Inception-v3.png?raw=true)

The resulting accuracy for the network on the test set was roughly 70.88%.
![](https://github.com/albert-zhong/dog-breed-identification/blob/main/Incepton-v3%20Accuracy.png?raw=true)



## Challenges
The main challenges came down to working with PyTorch to write clean, modularized code for plotting, training, and reshaping the output layer(s) of each network we intended to test against the dataset in question.

In particular, the use of Inception-v3 proved to require a fair bit of debugging due to the fact that Inception-v3 has an auxiliary output layer that we had failed to recognize when we were selecting networks to train. The usual flow for changing the output layer to reflect the number of labels obviously neglected that layer, and as such we needed to change our training code to grab those outputs in order to have the destination variable match the shape of the output coming from the network. Moreover, a separate set of preprocessing transforms needed to be defined as the input sizes to the network were 299x299 as opposed to the 227x227 that both ResNet models took in.

## Final Thoughts
In general, it's no surprise that the pretrained weights for general image recognition were able to perform well in a generic classification environment. In general, it seems that under controlled hyperparameters of training, deeper architectures such as ResNet-50 with more physical weights that affect classification yield better results. If we were to repeat such an experiment in the future, we'd definitely look into resources for longer training, where we free the weights on the pretrained networks and have gradient descent perturb the weights across the entirety of the networks.