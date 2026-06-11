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
