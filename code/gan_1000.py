# Imports
import tensorflow as tf
# import tensorflowjs as tfjs
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import glob
import time
from IPython import display
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("Testing if GPU is equipped with CUDA")
print(tf.test.is_built_with_cuda())

print("Listing all physical GPU devices")
print(tf.config.list_physical_devices('GPU'))

from lazylossbot.training_bot import from_txt, to_txt, automated_update, clear_txt, send_text

clear_txt()

# Get image dimensions from the files
def get_image_dimensions(path):
    if os.path.isdir(path + os.listdir(path)[0]):
        img = Image.open(path + os.listdir(path)[0])

    img = Image.open(path + os.listdir(path)[0])
    width, height = img.size
    return width, height

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


# -----------------------------------------------------------------------

# Global Variables

DATA_PATH = 'data/augmented/'
GENERATED_PATH = 'data/generated/'
SAVED_MODEL_PATH = 'data/saved_model/'
SAVED_MODEL_PATH_50 = 'data/saved_model/50/'

smallest_image_dim = get_smallest_image_dimensions(DATA_PATH)
image_dim = get_image_dimensions(DATA_PATH)
noise_dim = 1000
examples_to_generate = 1
random_seed = tf.random.normal([examples_to_generate, noise_dim])


# -----------------------------------------------------------------------

print("Loading data...")

if os.path.exists(DATA_PATH):
	print('Data found.')
else:
	print('No data found.')

# ! For tf <= 2.5 its 'tf.keras.preprocessing.image_dataset_from_directory', tf >= 2.5 its tf.keras.utils..
images = tf.keras.utils.image_dataset_from_directory(
  DATA_PATH,
  seed=42,
  image_size=image_dim,
  batch_size=16,
  label_mode=None,
  shuffle=True)

print("Normalizing data...")
data = images.map(lambda x: (x - 127.5) / 127.5)

print("Casting data...")
data = data.map(lambda x: tf.cast(x, tf.float32))



# Building the model
print("Initializing the model...")

# Generator class
class Generator(tf.keras.Model):
    """ Generator class """
    def __init__(self):
        super(Generator, self).__init__()

        self.g = tf.keras.Sequential()

        # Input random noise
        self.g.add(tf.keras.layers.Dense(16*16*256, use_bias=False, input_shape=(noise_dim,)))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Reshape to 2D
        self.g.add(tf.keras.layers.Reshape((16,16,256)))

        # Upsample to 32x32
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=1, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 64x64
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 128x128
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 256x256
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 512x512
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Output
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        self.g.add(tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Downsample
        self.g.add(tf.keras.layers.Conv2D(filters=4, kernel_size=11, strides=1, padding='valid', activation='relu'))
        self.g.add(tf.keras.layers.Conv2D(filters=4, kernel_size=8, strides=1, padding='valid', activation='relu'))
        self.g.add(tf.keras.layers.Conv2D(filters=4, kernel_size=8, strides=1, padding='valid', activation='relu'))

        # Output 224x224 image
        self.g.add(tf.keras.layers.Conv2D(filters=3, kernel_size=5, strides=1, padding='same', activation='tanh'))

    def call(self, inputs):
        return self.g(inputs)


# Discriminator class
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d = tf.keras.Sequential()

        # Input image
        self.d.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', input_shape=(image_dim[0], image_dim[1], 3)))
        self.d.add(tf.keras.layers.LeakyReLU())

        # Downsample to 56x56
        self.d.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same'))
        self.d.add(tf.keras.layers.LeakyReLU())
        self.d.add(tf.keras.layers.Dropout(0.1))

        # Downsample to 28x28
        self.d.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
        self.d.add(tf.keras.layers.LeakyReLU())
        self.d.add(tf.keras.layers.Dropout(0.1))

        # Downsample to 14x14, increase channels to 128
        self.d.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
        self.d.add(tf.keras.layers.LeakyReLU())
        self.d.add(tf.keras.layers.Dropout(0.1))

        # Downsample to 7x7, increase channels to 256
        self.d.add(tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same'))
        self.d.add(tf.keras.layers.LeakyReLU())
        self.d.add(tf.keras.layers.Dropout(0.1))

        # Flatten to 1D
        self.d.add(tf.keras.layers.Flatten())
        self.d.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output a 1 for a real image and a 0 for a generated image

    def call(self, inputs):
        return self.d(inputs)



# Generate images from noise
generated_image = Generator().call(random_seed) # Image is in range [-1, 1]

# Rescale image
generated_image = (generated_image.numpy() + 1) / 2

print(generated_image.shape)

# Set decision probability
decision = Discriminator().call(generated_image)

# Initializing loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Loss functions
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Initialize model
gen = Generator()
discriminator = Discriminator()

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([images.shape[0], 1000])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Generate images on noise
        generated_images = gen.call(noise)

        # Discriminator decision on real images and generated images
        real_output = discriminator.call(images)
        fake_output = discriminator.call(generated_images)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):

    predictions = model.call(test_input)

    img_to_show = (predictions[0].numpy() + 1) / 2
    img_to_show = np.reshape(img_to_show, (image_dim[0], image_dim[1], 3))
    plt.imsave(GENERATED_PATH + 'planet_at_epoch{:04d}.png'.format(epoch), img_to_show)

# Training function
def train(dataset, epochs):

    gen_losses = []
    disc_losses = []

    entire_time = time.time()

    send_text('Training starting...')

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        to_txt(epoch = epoch, loss = np.mean(gen_losses), acc = np.mean(disc_losses))

        if epoch % 50 == 0 and epoch > 1:
            print('Saving model.')
            input_arr = np.random.rand(1, 1000)
            outputs = gen(input_arr)
            gen.save_weights(SAVED_MODEL_PATH + 'generator_weights.h5')
            gen.save(SAVED_MODEL_PATH + 'generator_model')
            print('Model saved.')
            send_text('Model saved.')

        if epoch == 125:
            print('Saving 50% model.')
            input_arr = np.random.rand(1, 1000)
            outputs = gen(input_arr)
            gen.save_weights(SAVED_MODEL_PATH_50 + 'generator_weights.h5')
            gen.save(SAVED_MODEL_PATH_50 + 'generator_model')
            print('Model saved.')
            send_text('Model saved.')
        
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(gen, epoch + 1, random_seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(gen, epochs, random_seed)

    return gen_losses, disc_losses

# Training
print("Starting to train...")

gen_losses, disc_losses = train(data, 250)

print('Saving entire model...')

# Check if the path exists
if not os.path.exists(SAVED_MODEL_PATH):
    os.makedirs(SAVED_MODEL_PATH)

input_arr = np.random.rand(1, 1000)
outputs = gen(input_arr)
gen.save_weights(SAVED_MODEL_PATH + 'generator_weights.h5')
gen.save(SAVED_MODEL_PATH + 'generator_model')