import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model('model.h5')
