#Importing Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv
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


# Building and Training the model
# Reducing the training,testing loading images for neural network by which sys works properly
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Loading the Image_classifier.keras trained model
model = models.load_model('Image_classifier.keras')

img = cv.imread('car.jpg')                                 # Reading the testing image
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)                    # Enabling/Converting the BGR -> RGB colour

plt.imshow(img, cmap=plt.cm.binary)                         # cmap means: Color Map -> It is mainly used for grayscale images. Black pixels: 0 → Black & White pixels: 255 → White


predict = model.predict(np.array([img]) / 255)
index = np.argmax(predict)                                  # Find Largest Probability {np.argmax(predict)}, means: Return the index of the maximum value.

print(f"Prediction is {class_name[index]}")


plt.show()