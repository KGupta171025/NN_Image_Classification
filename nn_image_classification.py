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