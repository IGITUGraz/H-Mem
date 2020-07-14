"""Extracting layer that computes key and value."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class Extracting(Layer):
    """TODO"""

    def __init__(self,
                 units,
                 use_bias,
                 activation,
                 kernel_initializer,
                 kernel_regularizer,
                 **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        self.dense1 = Dense(units=self.units,
                            use_bias=self.use_bias,
                            activation=self.activation,
                            kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.kernel_regularizer)
        self.dense2 = Dense(units=self.units,
                            use_bias=self.use_bias,
                            activation=self.activation,
                            kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.kernel_regularizer)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=self.dtype)
            mask = tf.expand_dims(mask, axis=-1)
        else:
            mask = 1.0

        k = mask * self.dense1(inputs)
        v = mask * self.dense2(inputs)

        return tf.concat([k, v], axis=-1)

    def compute_mask(self, inputs, mask=None):
        return mask
