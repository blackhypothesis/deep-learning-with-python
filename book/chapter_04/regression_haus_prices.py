import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.datasets import california_housing


(train_data, train_targets), (test_data, test_targets) = (
    california_housing.load_data(version="small")
)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

x_train = (train_data - mean) / std
x_test = (test_data -mean) / std

y_train = train_targets / 100000
y_test = test_targets / 100000


def get_model():
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

k = 4
num_val_samples = len(x_train) // k
num_epochs = 200
all_mae_histories = []

for i in range(k):
    print(f"Processing fold #{i + 1}")
    fold_x_val = x_train[i * num_val_samples : (i + 1) * num_val_samples]
    fold_y_val = y_train[i * num_val_samples : (i + 1) * num_val_samples]
    fold_x_train = np.concatenate(
        [x_train[: i * num_val_samples], x_train[(i + 1) * num_val_samples :]],
        axis=0,
    )
    fold_y_train = np.concatenate(
        [y_train[: i * num_val_samples], y_train[(i + 1) * num_val_samples :]],
        axis=0,
    )

    model = get_model()
    history = model.fit(
        fold_x_train,
        fold_y_train,
        epochs=num_epochs,
        batch_size=16,
        verbose=0
    )

    mae_history = history.history["mean_absolute_error"]
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

epochs = range(1, len(average_mae_history) + 1)
plt.semilogy(epochs, average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation: mean absolute error")
plt.grid(which="major")
plt.show()
