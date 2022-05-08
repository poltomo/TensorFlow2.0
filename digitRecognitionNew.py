from this import d
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape)
#print(x_test.shape)

import numpy as np
import matplotlib.pyplot as plt

#print(np.min(x_train))
#print(np.max(x_train))

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()

x_train = x_train/255.0
x_test = x_test/255.0

model = keras.models.load_model("/home/leopard/Documents/TensorFlow2.0/theModel")

print(model.summary())

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=3)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss)
print(test_acc)

model.save("/home/leopard/Documents/TensorFlow2.0/theModel")
