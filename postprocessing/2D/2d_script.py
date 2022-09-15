
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import time
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
""" 

The mode was saved with .save (tf-function) and not with keras. 
This means that we can only load the model with tf.keras.models.load_model.

"""

# ------------------------------------------------------------
# Specify the path to the model

if len(sys.argv) > 1:
    path = sys.argv[1]
else: 
    path = 'data/saved_model/WGAN'

# Check if the path exists
if not os.path.exists('postprocessing/2D/temp/'):
    os.makedirs('postprocessing/2D/temp/')

if not os.path.exists('postprocessing/2D/result/'):
    os.makedirs('postprocessing/2D/result/')

# ------------------------------------------------------------
# Load the weights and generate an image

random_noise = tf.random.normal(shape=(1, 100))

start = time.time()

model = tf.keras.models.load_model(path, custom_objects={'tf': tf}, compile=False)
predictions = model(random_noise)
img_to_show = (predictions[0].numpy() + 1) / 2
img_to_show = np.reshape(img_to_show, (500, 500, 3))
plt.imsave('postprocessing/2D/temp/output.png', img_to_show)
print('Inference time: ', time.time() - start)

# ------------------------------------------------------------
# Grab image and apply blur / border

background = cv2.imread('postprocessing/2D/temp/output.png')
# background = cv2.GaussianBlur(background,(3,3),0) # Nice !
background = cv2.bilateralFilter(background,8,25,25) # Cool !


overlay = cv2.imread('postprocessing/2D/temp/background.png', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))

height, width = overlay.shape[:2]
for y in range(height):
    for x in range(width):
        overlay_color = overlay[y, x, :3]  # first three elements are color (RGB)
        overlay_alpha = overlay[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0

        # get the color from the background image
        background_color = background[y, x]

        # combine the background color and the overlay color weighted by alpha
        composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha

        # update the background image in place
        background[y, x] = composite_color

cv2.imwrite('postprocessing/2D/result/result.png', background)