#!/usr/bin/env python
"""Runs H-Mem on a single-shot image association task."""

import argparse
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import Model

from data.image_association_data import load_data
from layers.extracting import Extracting
from layers.reading import Reading
from layers.writing import WritingCell
from models.convnet14 import ConvNet14 as ConvNet

strategy = tf.distribute.MirroredStrategy()

parser = argparse.ArgumentParser()
parser.add_argument('--delay', type=int, default=0)
parser.add_argument('--timesteps', type=int, default=3)
parser.add_argument('--delay_padding', type=str, default='random', help='`zeros` or `random`')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size_per_replica', type=int, default=32)
parser.add_argument('--random_state', type=int, default=None)
parser.add_argument('--max_grad_norm', type=float, default=10.0)
parser.add_argument('--validation_split', type=float, default=0.1)

parser.add_argument('--retrain_convnet', type=int, default=0)
parser.add_argument('--use_pretrained_convnet', type=int, default=0)

parser.add_argument('--memory_size', type=int, default=200)
parser.add_argument('--dense_size', type=int, default=128)
parser.add_argument('--gamma_pos', type=float, default=0.01)
parser.add_argument('--gamma_neg', type=float, default=0.01)
parser.add_argument('--w_assoc_max', type=float, default=1.0)

parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size_per_replica * strategy.num_replicas_in_sync

# Set random seeds.
np.random.seed(args.random_state)
random.seed(args.random_state)
tf.random.set_seed(args.random_state)

# Load the data.
(x_train, y_train), (x_test, y_test) = load_data(timesteps=args.timesteps, merge=True,
                                                 data_dir='data/image_association_task/')

num_train = y_train.size - int(args.validation_split * y_train.size)
num_val = int(args.validation_split * y_train.size)
num_test = y_test.size

x_val = [x_train[0][-num_val:], x_train[1][-num_val:]]
y_val = y_train[-num_val:]

x_train = [x_train[0][:num_train], x_train[1][:num_train]]
y_train = y_train[:num_train]

timesteps_with_delay = args.timesteps * (args.delay + 1)

input_a_shape = (timesteps_with_delay, ) + x_train[0].shape[2:]
input_b_shape = x_train[1].shape[1:]


# Create the datasets.
def dataset_generator(x, y, seed):
    rng = np.random.RandomState(seed=seed)
    for a, b, y in zip(x[0], x[1], y):
        size = (timesteps_with_delay, ) + a.shape[1:-1] + (1,)
        if args.delay_padding == 'random':
            aa = rng.uniform(size=size).repeat(a.shape[-1], axis=3)
        elif args.delay_padding == 'zeros':
            aa = np.zeros(shape=size).repeat(a.shape[-1], axis=3)
        aa[::args.delay+1, :] = a

        yield {'input_a': aa, 'input_b': b}, y


output_types = ({'input_a': 'float32', 'input_b': 'float32'}, 'uint8')
output_shapes = ({'input_a': [None, None, None, None], 'input_b': [None, None, None]}, [])
train_dataset = tf.data.Dataset.from_generator(generator=lambda: dataset_generator(x_train, y_train, 42),
                                               output_types=output_types,
                                               output_shapes=output_shapes)
val_dataset = tf.data.Dataset.from_generator(generator=lambda: dataset_generator(x_val, y_val, 43),
                                             output_types=output_types,
                                             output_shapes=output_shapes)
test_dataset = tf.data.Dataset.from_generator(generator=lambda: dataset_generator(x_test, y_test, 44),
                                              output_types=output_types,
                                              output_shapes=output_shapes)

train_dataset = train_dataset.cache().repeat(args.epochs * batch_size).shuffle(10000).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Load pretrained model.
conv_net = ConvNet(include_top=False, name='conv_b')
if args.use_pretrained_convnet:
    conv_net.load_weights('saved_models/weights/ConvNet14-CIFAR10-MNIST/ConvNet14-CIFAR10-MNIST')
    conv_net.trainable = bool(args.retrain_convnet)

with strategy.scope():
    # Build the model.
    input_a = tf.keras.layers.Input(input_a_shape, name='input_a')
    input_b = tf.keras.layers.Input(input_b_shape, name='input_b')

    features_a = TimeDistributed(conv_net, name='conv_a')(input_a)
    features_a = TimeDistributed(tf.keras.layers.Flatten(), name='flatten_a')(features_a)
    features_a = TimeDistributed(tf.keras.layers.Dense(args.dense_size,
                                                       use_bias=False,
                                                       activation='relu',
                                                       kernel_initializer='he_uniform',
                                                       kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
                                 name='dense_a')(features_a)
    features_a = TimeDistributed(tf.keras.layers.BatchNormalization(), name='batch_norm_a')(features_a)
    features_a = TimeDistributed(tf.keras.layers.Dropout(0.3), name='dropout_a')(features_a)

    features_b = conv_net(input_b)
    features_b = tf.keras.layers.Flatten(name='flatten_b')(features_b)
    features_b = tf.keras.layers.Dense(args.dense_size,
                                       use_bias=False,
                                       activation='relu',
                                       kernel_initializer='he_uniform',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                       name='dense_b')(features_b)
    features_b = tf.keras.layers.BatchNormalization(name='batch_norm_b')(features_b)
    features_b = tf.keras.layers.Dropout(0.3, name='dropout_b')(features_b)

    entities = Extracting(units=args.memory_size,
                          use_bias=False,
                          activation='relu',
                          kernel_initializer='he_uniform',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                          name='entity_extracting')(features_a)

    memory_matrix = tf.keras.layers.RNN(WritingCell(units=args.memory_size,
                                                    gamma_pos=args.gamma_pos,
                                                    gamma_neg=args.gamma_neg,
                                                    w_assoc_max=args.w_assoc_max,
                                                    learn_gamma_pos=False,
                                                    learn_gamma_neg=False),
                                        name='entity_writing')(entities)

    queried_value = Reading(units=args.memory_size,
                            use_bias=False,
                            activation='relu',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                            name='entity_reading')(features_b, constants=[memory_matrix])

    outputs = tf.keras.layers.Dense(10,
                                    use_bias=False,
                                    kernel_initializer='he_uniform',
                                    name='output')(queried_value)

    model = Model(inputs=[input_a, input_b], outputs=outputs)

    # Compile the model.
    optimizer_kwargs = {'clipnorm': args.max_grad_norm} if args.max_grad_norm else {}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, **optimizer_kwargs),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model.summary()


# Train and evaluate.
def lr_scheduler(epoch):
    if epoch < 50:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (50 - epoch))


callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
model.fit(train_dataset,
          epochs=args.epochs,
          steps_per_epoch=np.ceil(num_train/batch_size),
          validation_data=val_dataset if num_val > 0 else None,
          validation_steps=np.ceil(num_val/batch_size),
          callbacks=[callback],
          verbose=args.verbose)

model.evaluate(test_dataset, steps=np.ceil(num_test/batch_size), verbose=2)
