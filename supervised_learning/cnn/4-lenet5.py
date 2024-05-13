#!/usr/bin/env python3
""" LeNet-5 (Tensorflow 1) """


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ A python function that builds a modified
    version of the LeNet-5 architecture using tensorflow """

    # Convolutional layer 1
    conv1 = tf.layers.Conv2D(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Max pooling layer 1
    pool1 = tf.layers.MaxPooling2D(
        inputs=conv1,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Convolutional layer 2
    conv2 = tf.layers.Conv2D(
        inputs=pool1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Max pooling layer 2
    pool2 = tf.layers.MaxPooling2D(
        inputs=conv2,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Flatten layer
    flat = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.Dense(
        inputs=flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Fully connected layer 2
    fc2 = tf.layers.Dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Output layer
    output = tf.layers.Dense(
        inputs=fc2,
        units=10,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Softmax activation
    softmax_output = tf.nn.softmax(output)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = y,
        logits = output
    )

    train_Adam = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)

    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    return softmax, train_Adam, loss, accuracy
