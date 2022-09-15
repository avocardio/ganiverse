
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
from skimage.metrics import structural_similarity as ssim

def new_loss(n, alpha):

    data_dir = 'data/generated/'

    if len(os.listdir(data_dir)) <= n:
        return 0

    last_5_images = os.listdir(data_dir)[-n:]

    for i in range(0,len(last_5_images)):
        last_5_images[i] = data_dir + last_5_images[i]

    images = np.array([np.asarray(Image.open(f).convert('RGB')) for f in last_5_images])

    ss = []

    for i in range(0,len(images)-1):
        ss.append(ssim(images[i], images[-1], multichannel=True))

    return alpha * np.mean(ss)