import tensorflow as tf
import pandas as pd
import matplotlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    #Creating fake data
    X = tf.random.normal(shape=[1000, 5])
    target = tf.random.uniform(minval=0, maxval=10, shape=(1000, 1), dtype=tf.int32)
    target = tf.cast(target, tf.float32)

    matrix = tf.concat([X, target], axis=1)

    window_size = 3
    dataset = tf.data.Dataset.from_tensor_slices(matrix )
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]))
    dataset = dataset.shuffle(1000).batch(10)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

    for X, Y in dataset.take(5):
        print(X, Y)