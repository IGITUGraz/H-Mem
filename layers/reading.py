"""Reading layers that read from memory."""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Layer


class Reading(Layer):
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

        self.dense = Dense(units=self.units,
                           use_bias=self.use_bias,
                           activation=self.activation,
                           kernel_initializer=self.kernel_initializer,
                           kernel_regularizer=self.kernel_regularizer)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, constants):
        memory_matrix = constants[0]

        k = self.dense(inputs)

        v = K.batch_dot(k, memory_matrix)

        return v

    def compute_mask(self, inputs, mask=None):
        return mask


class ReadingCell(Layer):
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

        self.dense = Dense(units=self.units,
                           use_bias=self.use_bias,
                           activation=self.activation,
                           kernel_initializer=self.kernel_initializer,
                           kernel_regularizer=self.kernel_regularizer)

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, states, constants):
        v = states[0]
        memory_matrix = constants[0]

        k = self.dense(tf.concat([inputs, v], axis=1))

        v = K.batch_dot(k, memory_matrix)

        return v, v

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros((batch_size, self.units))
