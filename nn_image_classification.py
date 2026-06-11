import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

print("=========================================")
print("  Environment Verified Successfully!  ")
print("=========================================")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version:      {np.__version__}")
print(f"Pandas version:     {pd.__version__}")
print(f"OpenCV version:     {cv2.__version__}")

# Check if GPU is available (in case TensorFlow is using CUDA/ROCm)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is available: {len(gpus)} device(s) found.")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("Running on CPU (No GPU detected or CUDA is not configured).")
print("=========================================")
