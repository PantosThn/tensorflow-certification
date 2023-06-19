import pandas as pd
import matplotlib as plt
import tensorflow as tf

def plot_series(time, series, format='-', start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlab('Time')
    plt.ylab('Value')
    plt.grid()

def sequence_to_sequence(series, window_size, batch_size, buffer_size=100):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=(window_size + 1), drop_remainder=True, shift=1)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    #window [1, 2, 3, 4] --> [2, 3, 4, 5] to become sequence to sequence for better flow of the gradient
    dataset = dataset.map(lambda window: (window[:-1], window[1:]), num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.cache().batch(batch_size).prefetch(AUTOTUNE)
    return dataset

if __name__ == "__main__":
    dataset = pd.read_csv('daily_min_temperatures.csv')

    train = dataset.iloc[:-365]
    test = dataset.iloc[-365:]

    train_len = int(len(dataset) * 0.8)
    train = dataset[:train_len]
    val = dataset[train_len:]

    train = train.iloc[:, -1]
    val = val.iloc[:, -1]
    test = test.iloc[:, -1]

    # Standardize the data
    mean = train.mean()
    std = train.std() + 1e-12
    train = (train-mean)/std
    val = (val-mean)/std
    test = (test-mean)/std

    '''
    label = dataset[3:]
    [1,2,3,4,5,6,7] if window is 3 then we will have the following
    [1,2,3] ---> 4
    [2,3,4] ---> 5
    [3,4,5] ---> 6
    [4,5,6] ---> 7
    [5,6,7] ---> x
    '''

    window_size = 30
    train_set = sequence_to_sequence(train.values, window_size=window_size, batch_size=128)
    val_set = sequence_to_sequence(val.values, window_size=window_size, batch_size=128)
    test_label = test.values[window_size:]

    # batch size, window size
    # (128, 30)

    for instance in train_set.take(1):
        print(instance)


