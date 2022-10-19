
""" This is the script for the custom SSIM-based loss function. """

# Imports
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
from skimage.metrics import structural_similarity as ssim


def new_loss(n, alpha): # Parameters: n = number of images in lookback, alpha = weight of SSIM loss

    data_dir = 'data/generated_256_1.8m/'

    # If the n images are not already generated, return 0
    if len(os.listdir(data_dir)) <= n:
        return 0

    # Get the last 5 images
    last_5_images = os.listdir(data_dir)[-n:]

    # Load the images
    for i in range(0,len(last_5_images)):
        last_5_images[i] = data_dir + last_5_images[i]

    # Load the images into a numpy array
    images = np.array([np.asarray(Image.open(f).convert('RGB')) for f in last_5_images])

    ss = []

    # Calculate the SSIM for each image in the lookback
    for i in range(0,len(images)-1):
        ss.append(ssim(images[i], images[-1], multichannel=True))

    # Return the average SSIM times the weight of the loss
    return alpha * np.mean(ss)