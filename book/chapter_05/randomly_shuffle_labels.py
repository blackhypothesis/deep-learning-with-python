import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.datasets import mnist


(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

random_train_labels = train_labels[:]

np.random.shuffle(random_train_labels)

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_images,
    random_train_labels,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
)

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

epochs = range(1, 101)
ax = plt.gca()
ax.set_ylim([0.1, 1.1])
plt.semilogy(
    epochs,
    accuracy,
    "r",
    label="Accuracy",
)
plt.semilogy(
    epochs,
    val_accuracy,
    "b--",
    label="Validation accuracy",
)
plt.title("Randomly shuffled labels")
plt.xlabel("Epochs")
plt.grid(which="major")
plt.legend()
plt.show()