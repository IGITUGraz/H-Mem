"""Word encodings."""

import numpy as np


def position_encoding(sentence_size, embedding_size):
    """Position Encoding.

    Encodes the position of words within the sentence (implementation based on
    https://arxiv.org/pdf/1503.08895.pdf [1]).

    Arguments:
      sentence_size: int, the size of the sentence (number of words).
      embedding_size: int, the size of the word embedding.

    Returns:
        A encoding matrix represented by a Numpy array with shape `[sentence_size, embedding_size]`.

    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size

    # Make position encoding of time words identity to avoid modifying them.
    encoding[:, -1] = 1.0

    return np.transpose(encoding)
