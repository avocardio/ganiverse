
""" This is the preprocessing script for the dataset. """

# Imports
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

# Get smallest image sizes
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
    data_dir = str(sys.argv[1])
except IndexError:
    data_dir = 'data/raw/*/*'

# Specify image size
try:
    img_size = (int(sys.argv[2]), int(sys.argv[2]))
except IndexError:
    img_size = get_smallest_image_dimensions(data_dir)

new_images = []

# Load images to memory
print('Loading images...')
for f in tqdm.tqdm(glob.iglob(data_dir)):
  new_images.append(np.asarray(Image.open(f)))

array_images = np.array(new_images, dtype=object)

print('Augmenting images...')

# Augment the images using regular augmentation techniques from the PIL module
def augment(input):

    path = 'data/augmented/'

    # create empty list
    augmented_images = []

    if os.path.exists(path):
        os.rmdir(path)
    os.mkdir(path)
    
    for idx, input in tqdm.tqdm((enumerate(input))):

        # 12 augmentation techniques in total

        # intialize counter for exact distribution of images into folders
        
        # 5 originals in a row, then new folder
        img = Image.fromarray(input)
        img = img.resize(img_size)
        # ------------
        
        # rotate image
        rot_img = img.rotate(90)
        rot_img = rot_img.save(f'{path}/{idx}_rotated.jpg')
        augmented_images.append(rot_img)

        # flip image top to bottom
        flip_img_tb = img.transpose(Image.FLIP_TOP_BOTTOM)
        flip_img_tb = flip_img_tb.save(f'{path}/{idx}_rotated_tb.jpg')
        augmented_images.append(flip_img_tb)

        # flip image top to bottom
        flip_img_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_img_lr = flip_img_lr.save(f'{path}/{idx}_rotated_lr.jpg')
        augmented_images.append(flip_img_lr)

        # colour conversion
        col_img = img.convert('L')
        col_img = col_img.save(f'{path}/{idx}_col.jpg')
        augmented_images.append(col_img)

        # blur image
        blur_img = img.filter(ImageFilter.BLUR)
        blur_img = blur_img.save(f'{path}/{idx}_blur.jpg')
        augmented_images.append(blur_img)

        # detail filter
        det_img = img.filter(ImageFilter.DETAIL)
        det_img = det_img.save(f'{path}/{idx}_detail.jpg')
        augmented_images.append(det_img)

        # enhance image
        enhanced_img = img.filter(ImageFilter.EDGE_ENHANCE)
        enhanced_img = enhanced_img.save(f'{path}/{idx}_enhanced.jpg')
        augmented_images.append(enhanced_img)
        
        # increase contrast
        enhancer_c = ImageEnhance.Contrast(img)
        contrast_img = enhancer_c.enhance(3)
        contrast_img = contrast_img.save(f'{path}/{idx}_contrast.jpg')
        augmented_images.append(contrast_img)

        # increase sharpness
        enhancer_s = ImageEnhance.Sharpness(img)
        sharp_img = enhancer_s.enhance(10.0)
        sharp_img = sharp_img.save(f'{path}/{idx}_sharp.jpg')
        augmented_images.append(sharp_img)

        # increase brightness
        enhancer_b = ImageEnhance.Brightness(img)
        bright_img = enhancer_b.enhance(2.0)
        bright_img = bright_img.save(f'{path}/{idx}_bright.jpg')
        augmented_images.append(bright_img)

    return augmented_images

# Main augmentation function
augmented = augment(array_images)

print('Total augmented images: ', len(augmented))