import tensorflow as tf
from tensorflow import keras

fashiondata = tf.keras.datasets.mnist # Loading DataSet
(x_train, y_train), (x_test, y_test) = fashiondata.load_data()

# Training Data
x_test.shape
x_train.shape

# Preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0

# Layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Optimization
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=5)