#!/usr/bin/env python3
# ^ to run scripts without specifying the interpreter

print("Testing imports")

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import glob
from IPython import display
import logging
import time
from telegram import Update, Bot, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters
import csv
import cv2
import skimage


print("Testing if GPU is equipped with CUDA")
print(tf.test.is_built_with_cuda())

print("Listing all physical GPU devices")
print(tf.config.list_physical_devices('GPU'))
