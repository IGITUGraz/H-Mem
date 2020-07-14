"""Writing layers that write to memory."""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Writing(Layer):
    """TODO"""

    def __init__(self,
                 units,
                 gamma,
                 learn_gamma=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self._gamma = gamma
        self.learn_gamma = learn_gamma

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,), trainable=self.learn_gamma,
                                     initializer=tf.keras.initializers.Constant(self._gamma),
                                     dtype=self.dtype, name='gamma')

        super().build(input_shape)

    def call(self, inputs, mask=None):
        k, v = tf.split(inputs, 2, axis=-1)

        k = tf.expand_dims(k, 2)
        v = tf.expand_dims(v, 1)

        hebb = self.gamma * k * v

        memory_matrix = hebb

        return memory_matrix

    def compute_mask(self, inputs, mask=None):
        return mask


class WritingCell(Layer):
    """TODO"""

    def __init__(self,
                 units,
                 gamma_pos,
                 gamma_neg,
                 w_assoc_max,
                 learn_gamma_pos=False,
                 learn_gamma_neg=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.w_max = w_assoc_max
        self._gamma_pos = gamma_pos
        self._gamma_neg = gamma_neg
        self.learn_gamma_pos = learn_gamma_pos
        self.learn_gamma_neg = learn_gamma_neg

    @property
    def state_size(self):
        return tf.TensorShape((self.units, self.units))

    def build(self, input_shape):
        self.gamma_pos = self.add_weight(shape=(1,), trainable=self.learn_gamma_pos,
                                         initializer=tf.keras.initializers.Constant(self._gamma_pos),
                                         dtype=self.dtype, name='gamma_pos')
        self.gamma_neg = self.add_weight(shape=(1,), trainable=self.learn_gamma_neg,
                                         initializer=tf.keras.initializers.Constant(self._gamma_neg),
                                         dtype=self.dtype, name='gamma_neg')

        super().build(input_shape)

    def call(self, inputs, states, mask=None):
        memory_matrix = states[0]
        k, v = tf.split(inputs, 2, axis=-1)

        k = tf.expand_dims(k, 2)
        v = tf.expand_dims(v, 1)

        hebb = self.gamma_pos * (self.w_max - memory_matrix) * k * v - self.gamma_neg * memory_matrix * k**2
        memory_matrix = hebb + memory_matrix

        return memory_matrix, memory_matrix

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros((batch_size, self.units, self.units))
