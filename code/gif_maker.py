from multiprocessing.dummy import current_process
import imageio.v2 as imageio
import os
import tensorflow_docs.vis.embed as embed
import glob
import sys

current_gifs = os.listdir('data/gifs/')

anim_file = f'data/gifs/gan_{len(current_gifs)+1}.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('data/generated/planet_at_epoch*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)