#!/usr/bin/env python3
""" Inception Network """


from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ A python function that builds the inception network
    as described in Going Deeper with Convolutions (2014) """

    image_input = K.Input(shape=(224, 224, 3))

    C1 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=(2, 2),
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_1 = C1(image_input)

    MP1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')
    output_2 = MP1(output_1)

    C2 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_3 = C2(output_2)

    MP2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')
    output_4 = MP2(output_3)

    I5 = inception_block(MP2, [64, 96, 128, 16, 32, 32])
    I6 = inception_block(I5, [128, 128, 192, 32, 96, 64])

    MP3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')
    output_5 = MP3(output_4)

    I7 = inception_block(MP3, [192, 96, 208, 16, 48, 64])
    I8 = inception_block(I7, [160, 112, 224, 24, 64, 64])
    I9 = inception_block(I8, [128, 128, 256, 24, 64, 64])
    I10 = inception_block(I9, [112, 144, 288, 32, 64, 64])
    I11 = inception_block(I10, [256, 160, 320, 32, 128, 128])

    MP4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')
    output_6 = MP4(output_5)

    I12 = inception_block(MP4, [256, 160, 320, 32, 128, 128])
    I13 = inception_block(I12, [384, 192, 384, 48, 128, 128])

    AP1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1),
                                    padding='same')
    output_7 = AP1(output_6)

    Dropout17 = K.layers.Dropout(rate=0.4)
    output_8 = Dropout17(output_7)

    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=K.initializers.he_normal())
    output_9 = output(output_8)

    model = K.Model(inputs=img_input, outputs=output)

    return model
