
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image, ImageOps
from PIL import ImageEnhance
from PIL import ImageFilter
import glob
from skimage.transform import resize
import tqdm
import sys

def get_smallest_image_dimensions(path):
    width = []
    height = []
    for i in os.listdir(path):
        # Check if path is nested
        if os.path.isdir(path + i):
            for j in os.listdir(path + i):
                img = Image.open(path + i + '/' + j)
                width.append(img.size[0])
                height.append(img.size[1])
            
        else:
            img = Image.open(path + i)
            width.append(img.size[0])
            height.append(img.size[1])

    return min(width), min(height)

# Paths
try:
    data_dir = sys.argv[1]
except IndexError:
    data_dir = 'data/raw/*/*'

try:
    number_of_mixups = sys.argv[2]
except IndexError:
    number_of_mixups = 1000

try:
    img_size = sys.argv[3]
except IndexError:
    img_size = get_smallest_image_dimensions(data_dir)



new_images = []

print('Loading images...')
#for f in tqdm.tqdm(glob.iglob(data_dir)):
#  new_images.append(np.asarray(Image.open(f)))

for f in tqdm.tqdm(os.listdir(data_dir)):
    new_images.append(np.asarray(Image.open(data_dir + f)))

array_images = np.array(new_images, dtype=object)

print('Mixing images...')

def mixup(array_images, alpha, n):

    mixup_images = []

    for n in tqdm.tqdm(range(n)):
        try: 
            img_1 = np.random.choice(array_images, replace=False)
            img_2 = np.random.choice(array_images, replace=False)
            
            if len(img_1.shape) < 3:
                img_1 = np.expand_dims(img_1, axis=2)
            if len(img_2.shape) < 3:
                img_2 = np.expand_dims(img_2, axis=2)

            img_1 = resize(img_1, img_size)
            img_2 = resize(img_2, img_size)

            mixed = img_1 * alpha + img_2 * (1 - alpha)
            
            img_image = Image.fromarray((255 * mixed).astype(np.uint8))
            img_image = img_image.save(f'data/augmented/{n+1}_mixup.jpg')

            mixup_images.append(mixed)
        except:
            pass
        
    mixup_images = np.asarray(mixup_images).astype(object)

    return mixup_images

mixup_images = mixup(array_images, 0.5, int(number_of_mixups))

print('Total mixup images: ', len(mixup_images))