#!/usr/bin/env python
"""Model for classifying CIFAR10 and MNIST jointly."""

import numpy as np
import tensorflow as tf

from models.convnet14 import ConvNet14
from utils.image_manipulation import expand_channels, pad

EPOCHS = 100
BATCH_SIZE = 128
PAD_WIDTH = ((2, 2), (2, 2), (0, 0))

# Load the data.
cifar10 = tf.keras.datasets.cifar10
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

mnist = tf.keras.datasets.mnist
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_mnist = expand_channels(x_train_mnist, num_channels=3)
x_test_mnist = expand_channels(x_test_mnist, num_channels=3)
x_train_mnist = pad(x_train_mnist, PAD_WIDTH)
x_test_mnist = pad(x_test_mnist, PAD_WIDTH)

# Concatenate MNISt and CIFAR10
x_train = np.concatenate((x_train_cifar10, x_train_mnist))
y_train = np.concatenate((y_train_cifar10.flatten(), y_train_mnist+10))
x_test = np.concatenate((x_test_cifar10, x_test_mnist))
y_test = np.concatenate((y_test_cifar10.flatten(), y_test_mnist+10))

idc = np.random.RandomState(seed=42).permutation(x_train.shape[0])
x_train = x_train[idc]
y_train = y_train[idc]
idc = np.random.RandomState(seed=42).permutation(x_test.shape[0])
x_test = x_test[idc]
y_test = y_test[idc]

input_shape = x_train.shape[1:]

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


def lr_scheduler(epoch):
    if epoch < 50:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (50 - epoch))


# Build the model.
model = ConvNet14(output_size=20)

# Compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train and evaluate the model
callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
          callbacks=[callback])

model.evaluate(x=x_test,  y=y_test, verbose=2)

# Save the models weights.
model.save_weights('saved_models/weights/ConvNet14-CIFAR10-MNIST/ConvNet14-CIFAR10-MNIST', save_format='tf')

model.summary()
