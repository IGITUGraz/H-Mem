"""Creates the data for the image association tasks."""

import os
import pathlib

import numpy as np
import tensorflow as tf

from utils import image_manipulation


def load_data(timesteps, pad_equal=False, merge=False, data_dir='data', seed=42):
    pad_equal = True if merge else pad_equal

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    suffix = '_{0}{1}.npy'.format(timesteps, '_merged' if merge else '')

    x_train_files = []
    x_test_files = []
    if merge:
        for i in ['_a', '_b']:
            x_train_files.append(pathlib.Path(data_dir, 'x_train' + i + suffix))
            x_test_files.append(pathlib.Path(data_dir, 'x_test' + i + suffix))
    else:
        for i in ['_a', '_b', '_c']:
            x_train_files.append(pathlib.Path(data_dir, 'x_train' + i + suffix))
            x_test_files.append(pathlib.Path(data_dir, 'x_test' + i + suffix))

    y_train_file = pathlib.Path(data_dir, 'y_train' + suffix)
    y_test_file = pathlib.Path(data_dir, 'y_test' + suffix)

    cifar10_train, cifar10_test = _get_cifar10_dataset()
    mnist_train, mnist_test = _get_mnist_dataset(num_channels=1)

    if not all([f.is_file() for f in x_train_files]) or not y_train_file.is_file():
        x_train, y_train = _combine_data(cifar10_train, mnist_train, pad_equal)
        x_train, y_train = _create_dataset(x_train, y_train, timesteps, merge, seed)

        for i in range(len(x_train_files)):
            np.save(x_train_files[i], x_train[i])
        np.save(y_train_file, y_train)

    x_train = []
    for i in range(len(x_train_files)):
        x_train.append(np.load(x_train_files[i], mmap_mode=None)[:12500])
    y_train = np.load(y_train_file, mmap_mode=None)[:12500]

    if not all([f.is_file() for f in x_test_files]) or not y_test_file.is_file():
        x_test, y_test = _combine_data(cifar10_test, mnist_test, pad_equal)
        x_test, y_test = _create_dataset(x_test, y_test, timesteps, merge, seed)

        for i in range(len(x_test_files)):
            np.save(x_test_files[i], x_test[i])
        np.save(y_test_file, y_test)

    x_test = []
    for i in range(len(x_test_files)):
        x_test.append(np.load(x_test_files[i], mmap_mode=None)[:2230])
    y_test = np.load(y_test_file, mmap_mode=None)[:2230]

    return (x_train, y_train), (x_test, y_test)


def _create_dataset(features, labels, timesteps, merge, seed):
    features_a, features_b = features

    num_classes = np.unique(labels).size
    shape_a = features_a.shape[1:]
    shape_b = features_b.shape[1:]

    features_a = np.reshape(features_a, (-1, num_classes) + shape_a, order='F')
    features_b = np.reshape(features_b, (-1, num_classes) + shape_b, order='F')
    labels = np.reshape(labels, (-1, num_classes), order='F')

    a1, a2 = np.split(features_a, (timesteps * len(features_a) // (timesteps+1), ))
    b1, b2 = np.split(features_b, (timesteps * len(features_b) // (timesteps+1), ))
    y1, y2 = np.split(labels, (timesteps * len(labels) // (timesteps+1), ))

    a1 = np.reshape(a1, (-1, timesteps) + shape_a)
    a2 = np.reshape(a2, (-1, ) + shape_a)
    b1 = np.reshape(b1, (-1, timesteps) + shape_b)
    y1 = np.reshape(y1, (-1, timesteps))
    y2 = np.reshape(y2, (-1, ))

    a1 = a1[:len(a1) // num_classes * num_classes]
    b1 = b1[:len(a1) // num_classes * num_classes]
    y1 = y1[:len(a1) // num_classes * num_classes]

    cols = []
    for i in range(y1.shape[1]):
        cols.append(np.unique(y1[:, i]))
    unique_cols = np.unique(cols, axis=0)

    x_a = a1
    x_b = b1

    # Targets are always the first `len(a1) // len(unique_cols)` elements in each timestep. We shuffle it
    # afterwards.
    y = -1 * np.ones(y1.shape[0], dtype=y1.dtype)
    x_c = np.zeros((a1.shape[0], ) + a2.shape[1:])
    target_mask = np.zeros(y1.shape, dtype=y1.dtype)
    for j, col in enumerate(unique_cols):
        for i in col:
            idc = np.nonzero(y1[:, j] == i)[0][:a1.shape[0] // num_classes]
            y[idc + j * a1.shape[0] // len(unique_cols)] = i

            target_mask[idc + j * a1.shape[0] // len(unique_cols), j] = 1

            idc2 = np.nonzero(y2 == i)[0][:a1.shape[0] // num_classes]
            x_c[idc + j * a1.shape[0] // len(unique_cols)] = a2[idc2]

    rows = np.split(np.indices(y1.shape)[0],
                    np.arange(num_classes // timesteps, len(y1), num_classes // timesteps))
    cols = np.split(np.indices(y1.shape)[1],
                    np.arange(num_classes // timesteps, len(y1), num_classes // timesteps))

    def shuffle_cols(cols, seed):
        return np.array([np.random.RandomState(seed=seed).permutation(c) for c in cols])

    def shuffle_rows(rows, seed):
        y = rows.shape[1]
        tmp = rows
        for i in range(y):
            tmp[:, i] = np.random.RandomState(seed=seed+i).permutation(rows[:, i])

        return tmp

    row_list_a, row_list_b = [], []
    col_list_a, col_list_b = [], []
    for i, (r, c) in enumerate(zip(rows, cols)):
        row_list_a.append(shuffle_rows(r, seed=seed*(i+1)))
        col_list_a.append(shuffle_cols(c, seed=seed*(i+2)))

        row_list_b.append(shuffle_rows(r, seed=seed*(i+3)))
        col_list_b.append(shuffle_cols(c, seed=seed*(i+4)))

    rows_a = np.concatenate(row_list_a)
    cols_a = np.concatenate(col_list_a)

    rows_b = np.concatenate(row_list_b)
    cols_b = np.concatenate(col_list_b)

    x_a = x_a[rows_a, cols_a]
    y_a = y1[rows_a, cols_a]
    if timesteps > 1:
        x_b = x_b[rows_b, cols_b]
        y_b = y1[rows_b, cols_b]
    else:
        y_b = y1
    x_c = np.stack([x_c] * timesteps, axis=1)
    x_c = x_c[rows_a, cols_a]

    target_mask = target_mask[rows_a, cols_a]
    x_c = x_c[np.nonzero(target_mask == 1)]
    y = y_b[np.nonzero(target_mask == 1)]

    idc = np.random.RandomState(seed=seed+10).permutation(y.shape[0])
    x_a = x_a[idc]
    x_b = x_b[idc]
    x_c = x_c[idc]
    y_a = y_a[idc]
    y_b = y_b[idc]
    y = y[idc]

    if merge:
        x_ab = []
        for a, b in zip(x_a.reshape((-1,) + x_a.shape[2:]), x_b.reshape((-1,) + x_b.shape[2:])):
            x_ab.append(image_manipulation.merge(a, b))

        x_ab = np.array(x_ab).reshape((-1, timesteps) + x_ab[0].shape)

        pad_width = ((0, 0), (0, x_c.shape[2]), (0, 0))
        x_c = image_manipulation.pad(x_c, pad_width=pad_width)

        return (x_ab, x_c), y
    else:
        return (x_a, x_b, x_c), y


def _combine_data(a, b, pad):
    features_a, labels_a = a
    features_b, labels_b = b

    labels_a = labels_a.flatten()
    labels_b = labels_b.flatten()

    num_classes_a = np.unique(labels_a).size
    num_classes_b = np.unique(labels_b).size
    min_num_examples_a = min(np.unique(labels_a, return_counts=True)[1])
    min_num_examples_b = min(np.unique(labels_b, return_counts=True)[1])

    assert num_classes_a == num_classes_b

    pad_width_dim1_a = pad_width_dim1_b = pad_width_dim2_a = pad_width_dim2_b = 0
    if pad:
        if features_a.shape[1] > features_b.shape[1]:
            pad_width_dim1_b = (features_a.shape[1] - features_b.shape[1]) // 2
        if features_a.shape[1] < features_b.shape[1]:
            pad_width_dim1_a = (features_b.shape[1] - features_a.shape[1]) // 2
        if features_a.shape[2] > features_b.shape[2]:
            pad_width_dim2_b = (features_a.shape[2] - features_b.shape[2]) // 2
        if features_a.shape[2] < features_b.shape[2]:
            pad_width_dim2_a = (features_b.shape[2] - features_a.shape[2]) // 2

    x_a = []
    x_b = []
    y = []
    for i in range(num_classes_a):
        idc_a = np.where(labels_a == i)[0]
        idc_b = np.where(labels_b == i)[0]
        num = min(idc_a.size, idc_b.size, min_num_examples_a, min_num_examples_b)
        x_a.append(features_a[idc_a[:num]])
        x_b.append(features_b[idc_b[:num]])
        y.append(labels_a[idc_a[:num]])

    x_a = np.concatenate(x_a)
    x_b = np.concatenate(x_b)
    y = np.concatenate(y)

    pad_width_a = ((pad_width_dim1_a, pad_width_dim1_a), (pad_width_dim2_a, pad_width_dim2_a), (0, 0))
    pad_width_b = ((pad_width_dim1_b, pad_width_dim1_b), (pad_width_dim2_b, pad_width_dim2_b), (0, 0))
    x_a = image_manipulation.pad(x_a, pad_width=pad_width_a)
    x_b = image_manipulation.pad(x_b, pad_width=pad_width_b)

    return (x_a, x_b), y


def _get_mnist_dataset(num_channels=1, pad_width=((0, 0), (0, 0), (0, 0))):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if num_channels == 1:
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    else:
        x_train = image_manipulation.expand_channels(x_train, num_channels=num_channels)
        x_test = image_manipulation.expand_channels(x_test, num_channels=num_channels)

    x_train = image_manipulation.pad(x_train, pad_width)
    x_test = image_manipulation.pad(x_test, pad_width)

    return (x_train, y_train), (x_test, y_test)


def _get_cifar10_dataset():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train.flatten()), (x_test, y_test.flatten())
