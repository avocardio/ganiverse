import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def importModel(modelPath):
    # model = tf.keras.models.load_model(modelPath)
    model = tf.keras.models.load_model(modelPath)
    print('Model loaded.')
    input_arr = tf.random.normal(shape=(1, 100))
    outputs = model(input_arr)
    print('Model inference.')
    tfjs.converters.save_keras_model(model, "tfjs")
    print('Successfully converted model to tfjs format')

importModel("data/saved_model/WGAN")