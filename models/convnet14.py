"""Convolutional network with 14 weight layers."""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

BatchNormalization._USE_V2_BEHAVIOR = False


class ConvNet14(Sequential):

    def __init__(self,
                 output_size=10,
                 include_top=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.output_size = output_size
        self.include_top = include_top

        weight_decay = 1e-3
        self.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

        self.add(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D((2, 2)))
        self.add(Dropout(0.1))

        self.add(Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D((2, 2)))
        self.add(Dropout(0.1))

        self.add(Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D((2, 2)))
        self.add(Dropout(0.2))

        self.add(Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D((2, 2)))
        self.add(Dropout(0.3))

        self.add(Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_uniform',
                 kernel_regularizer=self.kernel_regularizer, padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D((2, 2)))
        self.add(Dropout(0.3))
        if self.include_top:
            self.add(Flatten())
            self.add(Dense(512, activation='elu', kernel_initializer='he_uniform'))
            self.add(BatchNormalization())
            self.add(Dense(256, activation='elu', kernel_initializer='he_uniform'))
            self.add(BatchNormalization())
            self.add(Dense(128, activation='elu', kernel_initializer='he_uniform'))
            self.add(BatchNormalization())
            self.add(Dropout(0.4))
            self.add(Dense(self.output_size))
