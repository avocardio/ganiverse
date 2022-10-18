import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

random_seed = tf.random.normal([1, 100])

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

        # Upsample to 512x512
        self.g.add(tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=2, padding='same', use_bias=False))
        self.g.add(tf.keras.layers.BatchNormalization())
        self.g.add(tf.keras.layers.LeakyReLU())

        # A last downsampling to 500x500x3
        self.g.add(tf.keras.layers.Conv2D(filters=3, kernel_size=13, strides=1, padding='valid', use_bias=False, activation='tanh'))


    def call(self, inputs):
        return self.g(inputs)


# Discriminator class
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d = tf.keras.Sequential()

        # Input image
        self.d.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', input_shape=(500, 500, 3)))
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

gen = Generator()
disc = Discriminator()

# Generate images from noise
generated_image = gen.call(random_seed) # Image is in range [-1, 1]

gen.build(input_shape=(1, 100))
gen.summary()

# Rescale image
generated_image = (generated_image.numpy() + 1) / 2
print(generated_image.shape)

# Set decision probability
decision = disc.call(generated_image)