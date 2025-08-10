import tensorflow as tf
from tensorflow.experimental import numpy as tnp
import matplotlib.pyplot as plt

x = tf.Variable(initial_value=tf.range(-10, 10, 0.05, dtype=tf.float32))


def sigmoid(x):
    return tf.math.exp(x) / (1 + tf.math.exp(x))

def standard_normal_distribution(x):
    return tf.math.exp(-x*x) / 2 / tf.math.sqrt(2 * tnp.pi)

def friedrich_gauss(x):
    return tf.math.exp(-x*x) / tf.math.sqrt(tnp.pi)

with tf.GradientTape() as tape:
    y = sigmoid(x)

grad_y_wrt_x = tape.gradient(y, x)

plt.plot(x, y, "b", label="f(x)")
plt.plot(x, grad_y_wrt_x, "r", label="f'(x)")
plt.title("Gradient")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid("major")
plt.show()
