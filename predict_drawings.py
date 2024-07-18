import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model('model.h5')


#Create canvas
image = np.ones((400, 400), np.uint8) * 255
drawing = False # set to true on mouse press

ix, iy = -1, -1
# mouse callback
def draw_circle(event,x,y,flags,param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEDOWN:
        if drawing == True:
            cv2.line(image, (ix, iy), (x, y), (0, 0, 0), 4)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (ix, iy), (x, y), (0, 0, 0), 4)
