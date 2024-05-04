#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
train_mini_batch = __import__('3-mini_batch').train_mini_batch

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 5000

    np.random.seed(0)
    save_path = train_mini_batch(X_train, Y_train_oh, X_valid, Y_valid_oh,
                                 epochs=10, load_path='./graph.ckpt',
                                 save_path='./model.ckpt')
    print('Model saved in path: {}'.format(save_path))
ubuntu@alexa-ml:~/0x03-optimization$ ./3-main.py 
2018-11-10 02:10:48.277854: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 epochs:
    Training Cost: 2.8232288360595703
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810532331466675
    Validation Accuracy: 0.08640000224113464
    Step 100:
        Cost: 0.9012309908866882
        Accuracy: 0.6875
    Step 200:
        Cost: 0.6328266263008118
        Accuracy: 0.8125

    ...

    Step 1500:
        Cost: 0.27602481842041016
        Accuracy: 0.9375
After 1 epochs:
    Training Cost: 0.3164157569408417
    Training Accuracy: 0.9101600050926208
    Validation Cost: 0.291348934173584
    Validation Accuracy: 0.9168999791145325

...

After 9 epochs:
    Training Cost: 0.12963168323040009
    Training Accuracy: 0.9642800092697144
    Validation Cost: 0.13914340734481812
    Validation Accuracy: 0.961899995803833
    Step 100:
        Cost: 0.10656605660915375
        Accuracy: 1.0
    Step 200:
        Cost: 0.09849657118320465
        Accuracy: 1.0

    ...

    Step 1500:
        Cost: 0.0914708822965622
        Accuracy: 0.96875
After 10 epochs:
    Training Cost: 0.12012937664985657
    Training Accuracy: 0.9669600129127502
    Validation Cost: 0.13320672512054443
    Validation Accuracy: 0.9635999798774719
Model saved in path: