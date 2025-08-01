import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers

inputs = keras.Input(shape=(3,), name="input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_model")