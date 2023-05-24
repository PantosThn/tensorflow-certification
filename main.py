import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def univariate_window_dataset(series, window_size, batch_size, buffer_size=100):

    AUTOTUNE = tf.data.AUTOTUNE

    ''' from_tensors()

    returns: single element,
    type: TensorDataset

    from_tensor_slices()

    returns: multiple elements of input length,
    type: TensorSliceDataset '''

    dataset = tf.data.Dataset.from_tensor_slices(series)

    '''
    series = [0, 1, 2, 3, ..., 99]
    time = [0, 1, 2, 3, ..., 99]

    input = [0, 1, 2, ..., 9] ---> 10
    [10, 11, 12, ..., 19] ---> 20
    [20, 21, 22, ..., 29] ---> 30

    [1,2,3,4,5,6,7,8,9,10] ---> window size = 5, [1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7] shift = 1
    
    The window() method returns a dataset containing windows, where each window is itself represented as a dataset. 
    Something like {{1,2,3,4,5},{6,7,8,9,10},...}, where {...} represents a dataset. But we just want a regular dataset 
    containing tensors: {[1,2,3,4,5],[6,7,8,9,10],...}, where [...] represents a tensor. The flat_map() method returns 
    all the tensors in a nested dataset, after transforming each nested dataset. If we didn't batch, we would get: 
    {1,2,3,4,5,6,7,8,9,10,...}. By batching each window to its full size, we get {[1,2,3,4,5],[6,7,8,9,10],...} '''

    dataset = dataset.window(size=(window_size + 1), drop_remainder=True, shift=1)


    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]), num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)

    '''When the GPU is working on forward / backward propagation on the current batch, we want the CPU to process the 
    next batch of data so that it is immediately ready. As the most expensive part of the computer, we want the GPU to 
    be fully used all the time during training. We call this consumer / producer overlap, where the consumer is the GPU 
    and the producer is the CPU.With tf.data, you can do this with a simple call to dataset.prefetch(1) at the end of 
    the pipeline (after batching). This will always prefetch one batch of data and make sure that there is always one 
    ready.In some cases, it can be useful to prefetch more than one batch. For instance if the duration of the 
    preprocessing varies a lot, prefetching 10 batches would average out the processing time over 10 batches, instead of
     sometimes waiting for longer batches.'''
    dataset = dataset.cache().batch(batch_size).prefetch(AUTOTUNE)
    return dataset

if __name__ == '__main__':

    series = np.arange(5 * 365 + 1)
    print(series)
    dataset = univariate_window_dataset(series, window_size=10, batch_size=32)

    # Create train dataset and validation dataset
    split_time = 1000
    train = series[:split_time]
    validation = series[split_time:]

    train = univariate_window_dataset(train, window_size=30, batch_size=32)
    validation = univariate_window_dataset(validation, window_size=30, batch_size=32)

    tf.keras.backend.clear_session()

    #simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[30])
    ])

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer='adam', metrics=['mse'])
    history = model.fit(train, validation_data=validation, epochs=100)

    pd.DataFrame(history.history).plot(figsize=(10, 10))
    plt.grid()
    plt.show()
