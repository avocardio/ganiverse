import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mode = 1

path = 'data/saved_model/new/generator_model/'

model = tf.keras.models.load_model(path, custom_objects={'tf': tf}, compile=False)

random_noise = tf.random.normal(shape=(1, 100))
predictions = model(random_noise)
img_to_show = (predictions[0].numpy() + 1) / 2
img_to_show = np.reshape(img_to_show, (1000, 1000, 3))

plt.imsave('postprocessing/3D/temp/generated_image.jpg', img_to_show) # Needed to have a 4th dimension
img = plt.imread('postprocessing/3D/temp/generated_image.jpg')

if mode == 1:
    mesh = pv.Sphere()
    tex = pv.image_to_texture(img)

    mesh.texture_map_to_sphere(inplace=True)
    mesh.plot(texture=tex)
    mesh.save('postprocessing/3D/result/mesh.stl', texture=tex)

elif mode == 2:
    sphere = pv.Sphere(radius=1, theta_resolution=120, phi_resolution=120, start_theta=270.001, end_theta=270)
    sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))
    for i in range(sphere.points.shape[0]):
        sphere.active_t_coords[i] = [0.5 + math.atan2(-sphere.points[i, 0], sphere.points[i, 1])/(2 * math.pi), 0.5 + math.asin(sphere.points[i, 2])/math.pi]
    tex = pv.read_texture('postprocessing/3D/temp/generated_image.jpg')
    sphere.plot(texture=tex)
    sphere.save('postprocessing/3D/result/mesh.stl', texture=tex)