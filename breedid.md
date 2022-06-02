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

## Experiments and Evaluation

The full training and testing code we used to comapre the models can be viewed in our Colab notebook [here](https://colab.research.google.com/drive/1n4Donev0PE45W8-coGbfZ-s5n1rdc0x8?usp=sharing).

For each network we trained against the dataset, we generated plots for Training Loss vs. Epoch and Validation Loss vs. Epoch. We utilized a 70-15-15 split for training, validation, and testing.

The models we tested performed as follows:

### ResNet-34
![Training Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Training%20Loss%20vs%20Epoch%20-%20Modified%20Resnet-34.png?raw=true)

![Validation Loss vs. Epoch](https://github.com/albert-zhong/dog-breed-identification/blob/main/Validation%20Loss%20vs%20Epoch%20-%20Modified%20Resnet-34.png?raw=true)

The resulting accuracy for the network on the test set was roughly 79.17%.
![](https://github.com/albert-zhong/dog-breed-identification/blob/main/Resnet-34%20Accuracy.png?raw=true)


We define a function for summarizing the performance of the discriminator model. This function takes a sample of real galaxy images and generates the same number of fake galaxy images with the generator model and then evaluates the classification accuracy of the discriminator model and reports the score for each sample.

This is an example of the loss values on real images and fake generated images for the discriminator and the generator for epoch 26

    >26, 1/138, d1=0.29638204, d2=0.32488263 g=3.45844007
    >26, 2/138, d1=0.45061448, d2=0.88800186 g=4.71526003
    >26, 3/138, d1=1.16259015, d2=0.06006718 g=4.13229275
    >26, 4/138, d1=0.72051704, d2=0.09301022 g=2.78597784
    >26, 5/138, d1=0.81719440, d2=0.22286285 g=2.34299660 … … …
    >26, 134/138, d1=1.59510779, d2=0.09820783 g=2.37885666
    >26, 135/138, d1=0.35758907, d2=0.54601729 g=2.40457916
    >26, 136/138, d1=0.22124308, d2=0.10850765 g=2.61518002
    >26, 137/138, d1=0.18373565, d2=0.13180479 g=2.34858227
    >26, 138/138, d1=0.02594333, d2=0.20959115 g=2.79657221
    
This is an example of the model’s accuracy score for epoch 26 based on the discriminators ability to differentiate 

    >Accuracy real: 94%, fake: 100%
    

This is an example of the image produced by the model for epoch 26


![epoch 26](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e26.png)

## Results
Epoch 11

![epoch 11](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e11.png)

Epoch 12

![epoch 12](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e12.png)

Epoch 13

![epoch 13](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e13.png)

Epoch 14

![epoch 14](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e14.png)

Epoch 15

![epoch 15](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e15.png)

The results after the first training process for 15 epochs yielded the following results and we can see that the quality of the images are not consecutively increasing and from my understanding, Epoch 13 out of the last five epochs had the best result with the least amount of noise in comparison to the colorful galaxy center. At this stage the galaxy isn’t very detailed and more like a cloud of color, with a ton of noise and repetition of small sections, introducing patterns.

Epoch 41

![epoch 41](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e41.png)

Epoch 42

![epoch 42](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e42.png)

Epoch 43

![epoch 43](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e43.png)

Epoch 44

![epoch 44](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e44.png)

And finally...
Epoch 45

![epoch 45](https://raw.githubusercontent.com/inzombakura/inzombakura.github.io/main/assets/img/galaxygan_e45.png)

These are the results after the second training process for an additional 35 epochs. We can see that the quality does get better and there is more detail but we can see issues with repeated patterns in Epoch 41 and 42. The last three are fairly good representations of galaxies, where Epoch 43 has a very clean image, with a detailed galaxy center with little outside noise. Epoch 44 has a bit of a problem with the patterns of the outside of the galaxy with the red spots but the galaxy center is really rich in color and detail. Epoch 45 is the combination of the last two, where the center is colorful and clear and the surrounding is pretty clean with minimal noise.

## Final Thoughts
With more time we would definitely continue training this network, maybe on some persistent computer for at least a week. This would also allow us to attempt to generate original sized images at 256x256. It would also be wise for us to try new models like to add batch normalization layers to the layers of the discriminator which is quite easy with keras.

Thank you again to the whole team and to you for reading our report.