#!/usr/bin/env python3
""" LeNet-5 (Tensorflow 1) """


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ A python function that builds a modified
    version of the LeNet-5 architecture using tensorflow """

    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # build layer
    conv1 = tf.layers.Conv2D(inputs=x,
                             filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             kernel_initializer=initializer,
                             activation=tf.nn.relu,
                             name='conv1')
    pool1 = tf.layers.MaxPooling2D(inputs=conv1,
                                   pool_size=(2, 2),
                                   strides=(2, 2),
                                   name='pool1')
    conv2 = tf.layers.Conv2D(inputs=pool1,
                             filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             kernel_initializer=initializer,
                             activation=tf.nn.relu,
                             name='conv2')
    pool2 = tf.layers.MaxPooling2D(inputs=conv2,
                                   pool_size=(2, 2),
                                   strides=(2, 2),
                                   name='pool2')
    # flatten layers to convert tensor multidim in vector unidirectional
    flatten = tf.layers.Flatten(pool2)
    fc1 = tf.layers.Dense(inputs=flatten,
                          units=120,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer,
                          name='fc1')
    fc2 = tf.layers.Dense(inputs=fc1,
                          units=84,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer,
                          name='fc2')
    output = tf.layers.Dense(inputs=fc2,
                             units=10,
                             activation=None,
                             kernel_initializer=initializer,
                             name='logits')
    softmax = tf.nn.softmax(output)

    # calculate loss
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output)

    # Adam optimizer
    train_Adam = tf.train.AdamOptimizer().minimize(loss)

    # comparison of indice's max value for y and logits
    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)

    # convert tensor boll in float32
    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)

    # define accuracy
    accuracy = tf.reduce_mean(correct_prediction)

    return softmax, train_Adam, loss, accuracy
