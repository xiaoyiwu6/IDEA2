from keras.layers import Input, Conv2D, Concatenate, Add, Conv2DTranspose
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras


def ACblock(x, nb_filter, padding='same'):
    ACB_1 = Conv2D(nb_filter, (1, 3), padding=padding)(x)
    ACB_2 = Conv2D(nb_filter, (3, 3), padding=padding)(x)
    ACB_3 = Conv2D(nb_filter, (3, 1), padding=padding)(x)
    ACB = Add()([ACB_1, ACB_2])
    ACB = Add()([ACB, ACB_3])
    ACB = keras.layers.Activation('relu')(ACB)
    return ACB


def combine(x1, x2, nb_filter):
    M = Conv2D(nb_filter, (1, 1), padding='same', activation='relu')(x1)
    F = Conv2D(nb_filter, (1, 1), padding='same', activation='relu')(x2)
    comb = Concatenate(axis=3)([M, F])
    return comb


def net():
    inputs = Input((128, 128, 3))
    # extract features
    conv1 = Conv2D(32, (1, 1), padding='same')(inputs)
    conv1 = keras.layers.BatchNormalization(axis=3)(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    ACB1 = ACblock(conv1, 32, padding='same')
    ACB1 = ACblock(ACB1, 32, padding='same')
    pconv1 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(ACB1)

    ACB2 = ACblock(pconv1, 64, padding='same')
    ACB2 = ACblock(ACB2, 64, padding='same')
    pconv2 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(ACB2)

    ACB3 = ACblock(pconv2, 128, padding='same')
    ACB3 = ACblock(ACB3, 128, padding='same')
    pconv3 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(ACB3)

    ACB4 = ACblock(pconv3, 256, padding='same')
    ACB4 = ACblock(ACB4, 256, padding='same')
    pconv4 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same')(ACB4)

    ACB5 = ACblock(pconv4, 512, padding='same')
    ACB5 = ACblock(ACB5, 512, padding='same')
    # Context aggregation
    Context1 = Conv2D(512, (3, 3), padding='same', dilation_rate=1, activation='relu')(ACB5)
    Context2 = Conv2D(512, (3, 3), padding='same', dilation_rate=2, activation='relu')(ACB5)
    Context3 = Conv2D(512, (3, 3), padding='same', dilation_rate=5, activation='relu')(ACB5)

    Context1 = Conv2D(512, (1, 1), activation='relu', padding='same')(Context1)
    Context2 = Conv2D(512, (1, 1), activation='relu', padding='same')(Context2)
    Context3 = Conv2D(512, (1, 1), activation='relu', padding='same')(Context3)

    add1 = Add()([Context1, Context2])
    add1 = Add()([add1, Context3])
    add1 = keras.layers.Activation('relu')(add1)

    # upsampling
    deconv1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(add1)
    combine1 = combine(deconv1, ACB4, 256)
    ACB6 = ACblock(combine1, 256, padding='same')
    ACB6 = ACblock(ACB6, 256, padding='same')

    deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(ACB6)
    combine2 = combine(deconv2, ACB3, 128)
    ACB7 = ACblock(combine2, 128, padding='same')
    ACB7 = ACblock(ACB7, 128, padding='same')

    deconv3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(ACB7)
    combine3 = combine(deconv3, ACB2, 64)
    ACB8 = ACblock(combine3, 64, padding='same')
    ACB8 = ACblock(ACB8, 64, padding='same')

    deconv4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(ACB8)
    combine4 = combine(deconv4, ACB1, 32)
    ACB9 = ACblock(combine4, 32, padding='same')
    ACB9 = ACblock(ACB9, 32, padding='same')

    combine5 = combine(ACB9, conv1, 32)
    out = Conv2D(1, (1, 1), activation='sigmoid')(combine5)

    model = Model(inputs, out)

    return model
