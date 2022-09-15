# Imports
import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres
from PIL import Image
import sys
import glob
import os
import numpy as np
import tqdm
import time

image_size = (500,500)

start = time.time()

multiplicator = int(sys.argv[2])

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the model
model_path = f'Postprocessing/SR_Model/EDSR_x{multiplicator}.pb'
sr.readModel(model_path)

print('Loading model:', time.time() - start)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", multiplicator)

image_path = sys.argv[1]

if 'raw' in image_path:
    if not os.path.exists('data/raw_sr'):
        os.mkdir('data/raw_sr')
    store_path = 'data/raw_sr'

if 'augmented' in image_path:
    if not os.path.exists('data/augmented_sr'):
        os.mkdir('data/augmented_sr')
    store_path = 'data/augmented_sr'

if 'generated' in image_path:
    if not os.path.exists('data/generated_sr'):
        os.mkdir('data/generated_sr')
    store_path = 'data/generated_sr'

else:
    os.mkdir('data/generated_sr')
    store_path = 'data/generated_sr'


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, image_size)

# Upscale image
start = time.time()
image = sr.upsample(image)
print('Upsample:', time.time() - start)
plt.imsave(f'{store_path}/image_sr.png', image)

print("SR done!")