#Importing Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras import datasets, layers, models

# CIFAR-10 images are loaded as uint8 (0-255).
# Neural networks perform calculations using decimal values, so we convert
# images to float32 and normalize pixel values to the range [0, 1].
# float32 is preferred because it supports decimal computations while using
# less memory and training faster than float64.
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images.astype('float32') / 255.0, testing_images.astype('float32') / 255.0


class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']



# Display the first 16 training images in a 4x4 grid.
# Note:
#   training_labels contains class IDs, NOT sequential numbers.
#   Example first few labels may be [6, 9, 9, 4, 1, ...]
#   because the dataset is shuffled/mixed, not sorted by class.

# subplot works: {plt.subplot(4,4,i+1)}
# -4 rows
# -4 columns
# -position = i + 1

# 1   2   3   4
# 5   6   7   8
# 9  10  11  12
# 13 14  15  16

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[training_labels[i][0]])


plt.show()

# training_labels[0]      -> [6]
# training_labels[0][0]   -> 6
# class_name[6]           -> "Frog"

# Dataset labels are mixed (6,9,9,4,1,...) because the dataset
# is not sorted by class. Mixed data helps neural networks learn better.


# Building and Training the model
# Reducing the training,testing loading images for neural network by which sys works properly
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


# CNN Pipeline:-

# Create an empty CNN model where layers are added sequentially.
# model = models.Sequential()
# Think of it like:
# Input
#  ↓
# Layer 1
#  ↓
# Layer 2
#  ↓
# Layer 3
#  ↓
# Output
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',input_shape = (32,32,3)))      # First convolution layer:- Input image: 32x32x3, Learn 32 feature maps using 3x3 filter & ReLU keeps positive values and sets negatives to 0.
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))                              # Second Layer
model.add(layers.MaxPooling2D((2,2)))                           # Downsample feature maps using 2x2 pooling. Keeps strongest features while reducing image size.
model.add(layers.Conv2D(64, (3,3), activation='relu'))                              # Third Layer
model.add(layers.Flatten())                                     # Shape: (4,4,64) → (1024,) or Before: 4 × 4 × 64 → Total values: 4 × 4 × 64 = 1024
model.add(layers.Dense(64,activation='relu'))                   # Every neuron connects to every neuron in previous layer → Fully connected layer with 64 neurons for feature learning.
model.add(layers.Dense(10,activation='softmax'))                # Output layer: 10 neurons (one per CIFAR-10 class) & Softmax converts outputs into class probabilities.
# input_shape=(32,32,3) → CIFAR-10 images are: Height = 32, Width  = 32 & Channels = 3 (RGB); So: (32,32,3) → means: 32 × 32 pixels & 3 color channels

'''
Why Dense → 10? with softmax activation
Because CIFAR-10 has:
Plane
Car
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
(10 classes)

For Dense: Softmax:-
Converts outputs into probabilities.
Example:

Plane  = 0.01
Car    = 0.03
Bird   = 0.02
Cat    = 0.01
Deer   = 0.05
Dog    = 0.02
Frog   = 0.04
Horse  = 0.02
Ship   = 0.10
Truck  = 0.70

Total: 1.00
Prediction: Truck (70%)
'''


'''
Complete Flow:

32x32x3 Image
      ↓
Conv2D(32)
      ↓
30x30x32
      ↓
MaxPool
      ↓
15x15x32
      ↓
Conv2D(64)
      ↓
13x13x64
      ↓
MaxPool
      ↓
6x6x64
      ↓
Conv2D(64)
      ↓
4x4x64
      ↓
Flatten
      ↓
1024
      ↓
Dense(64)
      ↓
Dense(10)
      ↓
Class Probabilities


This is the standard CNN pipeline:

Image
 ↓
Feature Extraction (Conv + Pool)
 ↓
Flatten
 ↓
Classification (Dense Layers)
 ↓
Predicted Class
'''
