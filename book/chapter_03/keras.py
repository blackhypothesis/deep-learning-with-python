import os
os.environ["KERAS_BACKEND"] = "jax"

# from keras import layers
# import keras

# layer = layers.Dense(32, activation="relu")

# print(layer)

# class SimpleDense(keras.Layer):
#     def __init__(self, units, activation=None):
#         super().__init__()
#         self.units = units
#         self.activation = activation

#     def build(self, input_shape):
#         batch_dim, input_dim  = input_shape
#         self.W = self.add_weight(
#             shape=(input_dim, self.units), initializer="random_normal"
#         )
#         self.b = self.add_weight(shape=(self.units,), initializer="zeros")

#     def call(self, inputs):
#         y = keras.ops.matmul(input, self.W) + b
#         if self.activation is not None:
#             y = self.activation(y)
#         return y

# my_dense = SimpleDense(units=32, activation=keras.ops.relu)
# input_tensor = keras.ops.ones(shape=(2, 784))
# output_tensor = my_dense(input_tensor)
# print(output_tensor.shape)

from keras import models
from keras import layers
model = models.Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(32),
    ]
)