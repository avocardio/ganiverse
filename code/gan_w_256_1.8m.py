
""" This is the main training file for the WGAN. """

# Imports
import tensorflow as tf
from keras import backend
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import glob
import time
from IPython import display
import sys
from ssim_loss import new_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Test if cuda is working
print("Testing if GPU is equipped with CUDA")
print(tf.test.is_built_with_cuda())

print("Listing all physical GPU devices")
print(tf.config.list_physical_devices('GPU'))

# Import the Telegram update bot
from lazylossbot.training_bot import from_txt, to_txt, automated_update, clear_txt, send_text

# Clear the temporary loss text file for the bot
clear_txt()

# Function for getting image dimensions from a path
def get_image_dimensions(path):
    if os.path.isdir(path + os.listdir(path)[0]):
        img = Image.open(path + os.listdir(path)[0])

    img = Image.open(path + os.listdir(path)[0])
    width, height = img.size
    return width, height

# Function for getting the smallest image dimensions from a path
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
GENERATED_PATH = 'data/generated_256_1.8m/'
SAVED_MODEL_PATH = 'data/saved_model/256_1.8m/'
SAVED_MODEL_PATH_50 = 'data/saved_model/256_1.8m/50/'

if os.path.exists(GENERATED_PATH):
    os.rmdir(GENERATED_PATH)
os.mkdir(GENERATED_PATH)

# image_dim = get_smallest_image_dimensions(DATA_PATH)
image_dim = (256,256) # We know the data is 500,500,3
noise_dim = 100 # Noise dimension for the generator
examples_to_generate = 1 # Number of images to generate for saving / showing progress via the bot
random_seed = tf.random.normal([examples_to_generate, noise_dim]) # Random noise for generaor


# -----------------------------------------------------------------------

# Preprocessing pipeline

print("Loading data...")

if os.path.exists(DATA_PATH):
	print('Data found.')
else:
	print('No data found.')

# ! For tf <= 2.5 its 'tf.keras.preprocessing.image_dataset_from_directory', tf >= 2.5 its tf.keras.utils..
images = tf.keras.utils.image_dataset_from_directory(
  DATA_PATH,
  seed=42,
  image_size=(256,256),
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
        self.g.add(tf.keras.layers.Dense(16*16*64, use_bias=False, input_shape=(100,)))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Reshape to 2D
        self.g.add(tf.keras.layers.Reshape((16,16,64)))

        # Upsample to 32x32
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 64x64
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 128x128
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # Upsample to 256x256
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        self.g.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False, activation='tanh'))


    def call(self, inputs):
        return self.g(inputs)

# Discriminator class
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d = tf.keras.Sequential()

        # Input image
        self.d.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', input_shape=(256, 256, 3)))
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
        self.d.add(tf.keras.layers.Dense(1)) 

    def call(self, inputs):
        return self.d(inputs)

# Generate images from noise
generated_image = Generator().call(random_seed) # Image is in range [-1, 1]

# Rescale image
generated_image = (generated_image.numpy() + 1) / 2

# Set decision probability
decision = Discriminator().call(generated_image)

# -----------------------------------------------------------------------

# Loss functions

# Loss function for the discriminator (critic): the mean absolute difference between the generated and the real image samples
def wasserstein_loss_critic(real_output, fake_output):
    return backend.mean(real_output) - backend.mean(fake_output)

# Loss function for the generator: mean of our generated images multiplied by our custom SSIM loss, with a lookback of 3 and scaled by 1
def wasserstein_loss_generator(fake_output):
    return backend.mean(fake_output) * (1 + new_loss(3,1))

# We compute the gradient penalty as an additional term to the critic loss
def gradient_penalty(real_output, fake_output):
    alpha = tf.random.uniform(shape=[real_output.shape[0], 1, 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real_output + (1 - alpha) * fake_output
    gradients = tf.gradients(discriminator.call(interpolated), [interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    return gradient_penalty

# Optimizers
generator_optimizer = RMSprop(learning_rate=0.0001, decay=0.00001)
discriminator_optimizer = RMSprop(learning_rate=0.0001, decay=0.00001)

# Initialize model
gen = Generator()
discriminator = Discriminator()

# -----------------------------------------------------------------------

# Training step

@tf.function
def train_step(images):
    # Generate noise in the shape of the image batch
    noise = tf.random.normal([images.shape[0], noise_dim])

    # For every training iteration, we call the critic 3 times
    for _ in range(2):
        with tf.GradientTape() as disc_tape:
            generated_image = gen.call(noise)
            real_output = discriminator.call(images)
            fake_output = discriminator.call(generated_image)
            disc_loss = wasserstein_loss_critic(real_output, fake_output)

            gp = gradient_penalty(images, generated_image) # Calculate gradient penalty
            disc_loss += 1.5 * gp # Add gradient penalty to the loss with a weight of 1.5

        # Backpropagate the gradients
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # For every training iteration, we call the generator once
    with tf.GradientTape() as gen_tape:
        generated_image = gen.call(noise)
        fake_output = discriminator.call(generated_image)
        gen_loss = wasserstein_loss_generator(fake_output)

    # Backpropagate the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))

    # Clip the weights
    for weight in gen.trainable_variables:
        weight.assign(tf.clip_by_value(weight, -0.5, 0.5))

    for weight in discriminator.trainable_variables:
        weight.assign(tf.clip_by_value(weight, -0.5, 0.5))

    return gen_loss, disc_loss

# Function to save output images to a folder, for visualization and for the bot 
def generate_and_save_images(model, epoch, test_input):
    if not os.path.exists(GENERATED_PATH):
        os.makedirs(GENERATED_PATH)

    predictions = model.call(test_input)

    img_to_show = (predictions[0].numpy() + 1) / 2
    img_to_show = np.reshape(img_to_show, (image_dim[0], image_dim[1], 3))
    plt.imsave(GENERATED_PATH + 'planet_at_epoch{:04d}.png'.format(epoch), img_to_show)

# Training function
def train(dataset, epochs):

    gen_losses = []
    disc_losses = []

    entire_time = time.time()

    # send_text('Training starting...')

    # Main training loop function
    for epoch in range(epochs):
        start = time.time()

        # For every image batch in the dataset, we train the model and append losses to the lists
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        # The mean losses are sent to the bot for monitoring
        to_txt(epoch = epoch, loss = np.mean(gen_losses), acc = np.mean(disc_losses))

        # Every 50 epochs, we save the model weights in the tf format
        if epoch % 50 == 0 and epoch > 1:
            print('Saving model.')
            input_arr = np.random.rand(1, noise_dim)
            outputs = gen(input_arr)
            gen.save(SAVED_MODEL_PATH + 'generator_model', save_format='tf')
            print('Model saved.')
            send_text('Model saved.')

        # At half the epochs, we save the model weights again
        if epoch == 250:
            print('Saving 50% model.')
            input_arr = np.random.rand(1, noise_dim)
            outputs = gen(input_arr)
            gen.save(SAVED_MODEL_PATH_50 + 'generator_model', save_format='tf')
            print('Model saved.')
            send_text('Model saved.')
        
        # Save the images every epoch
        display.clear_output(wait=True)
        generate_and_save_images(gen, epoch + 1, random_seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(gen, epochs, random_seed)

    return gen_losses, disc_losses

# -----------------------------------------------------------------------

# Training

print("Starting to train...")

# Train the model for 500 epochs
gen_losses, disc_losses = train(data, 1000)

send_text('Training finished.')

print('Saving entire model...')

# Check if the path exists
if not os.path.exists(SAVED_MODEL_PATH):
    os.makedirs(SAVED_MODEL_PATH)

# Save the entire model one last time with the final weights
input_arr = np.random.rand(1, noise_dim)
outputs = gen(input_arr)
gen.save(SAVED_MODEL_PATH + 'generator_model', save_format='tf')
