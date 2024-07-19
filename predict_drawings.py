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

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(image, (ix, iy), (x, y), (0, 0, 0), 4)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (ix, iy), (x, y), (0, 0, 0), 4)

cv2.namedWindow('draw')
cv2.setMouseCallback('draw', draw_circle)

# main While Loop

while True:
    cv2.imshow('draw', image)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        image = np.ones((400, 400), np.uint8) * 255
    elif k == ord('q'):
        break
    elif k == ord('p'):
        resized = cv2.resize(image, (28, 28))

        resized = cv2.bitwise_not(resized)

        resized = resized / 255.0

        input_image = resized.reshape(1, 28, 28)

        prediction = model.predict(input_image)
        predicted_digit = np.argmax(prediction)
        print(f"PREDICTION: {predicted_digit}")

cv2.destroyAllWindows()
