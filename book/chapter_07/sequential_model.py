import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers

model = keras.Sequential(name="my_model")
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu", name="first_layer"))
model.add(layers.Dense(10, activation="softmax", name="output_layer"))

# model.build(input_shape=(None, 3))
print(model.weights)
print(model.summary())