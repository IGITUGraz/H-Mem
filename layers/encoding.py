"""Sentence encoding."""

import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer

from utils.word_encodings import position_encoding


class Encoding(Layer):
    """TODO"""

    def __init__(self,
                 encodings_type,
                 encodings_constraint,
                 **kwargs):
        super().__init__(**kwargs)

        self.encodings_type = encodings_type.lower()
        self.encodings_constraint = encodings_constraint.lower()

        if self.encodings_type not in ('identity_encoding', 'position_encoding', 'learned_encoding'):
            raise ValueError('Could not interpret encodings type:', self.encodings_type)

        if self.encodings_constraint not in ('none', 'mask_time_word'):
            raise ValueError('Could not interpret encodings constraint:', self.encodings_type)

        self.constraint = self.MaskTimeWord() if self.encodings_constraint == 'mask_time_word' else None

    def build(self, input_shape):
        if self.encodings_type.lower() == 'identity_encoding':
            self.encoding = tf.ones((input_shape[-2], input_shape[-1]))
        if self.encodings_type.lower() == 'position_encoding':
            self.encoding = position_encoding(input_shape[-2], input_shape[-1])
        if self.encodings_type.lower() == 'learned_encoding':
            self.encoding = self.add_weight(shape=(input_shape[-2], input_shape[-1]), trainable=True,
                                            initializer=tf.initializers.Ones(),
                                            constraint=self.constraint,
                                            dtype=self.dtype, name='encoding')

        super().build(input_shape)

    def call(self, inputs, mask=None):
        mask = tf.cast(mask, dtype=self.dtype)
        mask = tf.expand_dims(mask, axis=-1)

        return tf.reduce_sum(mask * self.encoding * inputs, axis=-2)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        return tf.reduce_any(mask, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    class MaskTimeWord(Constraint):
        """Make encoding of time words identity to avoid modifying them."""

        def __init__(self,
                     **kwargs):
            super().__init__(**kwargs)

        def __call__(self, w):
            indices = [[w.shape[0]-1]]
            updates = tf.ones((1, w.shape[1]))
            new_w = tf.tensor_scatter_nd_update(w, indices, updates)

            return new_w
