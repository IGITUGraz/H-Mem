"""Image manipulations."""

import numpy as np


def merge(a, b):
    """Merge two images to one.

    The images are stacked column wise (left `a` and right `b`).

    Arguments:
      a: iterable, The images.
      b: iterable, The images.

    Returns:
        A Numpy array containing the merged images.

    """
    rows_a, cols_a, channels_a = a.shape
    rows_b, cols_b, channels_b = b.shape

    rows = max(rows_a, rows_b)
    cols = cols_a + cols_b
    channels = max(channels_a, channels_b)

    c = np.zeros(shape=(rows, cols, channels))
    c[:rows_a, :cols_a] = a
    c[:rows_b, cols_a:] = b

    return c


def pad(x, pad_width):
    """Pad images in `x` with zeros.

    Arguments:
      x: iterable, The images to pad.
      pad_width: sequence, array_like, int, Number of values padded to the edges of each axis. ((before_1,
       after_1), … (before_N, after_N)) unique pad widths for each axis. ((before, after),) yields same before
       and after pad for each axis. (pad,) or int is a shortcut for before = after = pad width for all axes.

    Returns:
        A Numpy array containing the padded images.

    """
    y = []
    for item in x:
        y.append(np.pad(item, pad_width=pad_width))

    y = np.array(y)

    return y


def expand_channels(x, num_channels=3):
    y = []
    for i, item in enumerate(x):
        rows, cols = item.shape
        c = np.zeros(shape=(rows, cols, num_channels))
        c[:, :, :] = item[:, :, np.newaxis]
        y.append(c)

    y = np.array(y)

    return y
