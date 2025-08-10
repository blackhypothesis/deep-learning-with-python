import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import tensorboard
from keras import layers
from keras.datasets import mnist

def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(64, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model


(images, labels), (test_images, test_labels) = mnist.load_data()

images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

tb = keras.callbacks.TensorBoard(
    log_dir="/deep-learning/book/chapter_07/log",
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=[tb],
    validation_data=(val_images, val_labels),
)
