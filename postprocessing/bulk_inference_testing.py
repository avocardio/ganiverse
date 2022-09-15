import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
""" 

The mode was saved with .save (tf-function) and not with keras. 
This means that we can only load the model with tf.keras.models.load_model.

Alternatives: 

model.load_weights(path).expect_partial()
tf.train.Checkpoint.restore(checkpoint, path).expect_partial()

Better: saver = tf.train.Saver(tf.model_variables())

"""

if len(sys.argv) > 1:
    path = sys.argv[1]
else: 
    path = 'data/saved_model/new/generator_model'

start = time.time()

model = tf.keras.models.load_model(path, custom_objects={'tf': tf}, compile=False)

plt.figure(figsize=(10, 5), dpi=120)
for i in range(10):
    random_noise = tf.random.normal(shape=(1, 100))
    predictions = model(random_noise)
    img_to_show = (predictions[0].numpy() + 1) / 2
    img_to_show = np.reshape(img_to_show, (1000, 1000, 3))
    plt.subplot(2, 5, i + 1)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.imshow(img_to_show)
    plt.axis('off')
    
print('Inference time: ', time.time() - start)
plt.show()