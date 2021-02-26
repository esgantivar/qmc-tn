import tensorflow as tf
import numpy as np


def binary_map(x, v_max=255.0):
    n_data, dim0, dim1 = tuple(x.shape)
    n_sites = dim0 * dim1
    x = x.reshape((n_data, n_sites)) / v_max
    x = tf.cast(tf.math.greater(x, 0.5), dtype=tf.int32)
    return tf.keras.utils.to_categorical(y=x, num_classes=2)


def binary_map_p(x, v_max=255.0):
    n_data, dim0, dim1 = tuple(x.shape)
    n_sites = dim0 * dim1
    x = x.astype(np.float32).reshape((n_data, n_sites)) / v_max
    return np.stack([x, 1 - x], axis=2)


def map_stoudenmire(x, v_max=255.0):
    n_data, dim0, dim1 = tuple(x.shape)
    n_sites = dim0 * dim1
    x = (x.reshape((n_data, n_sites)) / v_max) * (np.pi / 2)
    sin = tf.math.sin(x)
    cos = tf.math.cos(x)
    return tf.cast(tf.stack([cos, sin], axis=2), dtype=tf.float32)
