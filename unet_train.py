import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import keras
from keras.layers.merge import concatenate
from keras.preprocessing.image import img_to_array
import random
import os
import matplotlib.pyplot as plt
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_w = 256
img_h = 256

n_label = 1


def unet():
    inputs = Input((img_w, img_h, 3))

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def another_unet():
    inputs = Input((img_w, img_h, 3))

    conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = keras.layers.Activation('elu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = keras.layers.Activation('elu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = keras.layers.Activation('elu')(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = keras.layers.Activation('elu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = keras.layers.Activation('elu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = keras.layers.Activation('elu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(pool3)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = keras.layers.Activation('elu')(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = keras.layers.Activation('elu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_uniform')(pool4)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = keras.layers.Activation('elu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = keras.layers.Activation('elu')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(up6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = keras.layers.Activation('elu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = keras.layers.Activation('elu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(up7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = keras.layers.Activation('elu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = keras.layers.Activation('elu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(up8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = keras.layers.Activation('elu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = keras.layers.Activation('elu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(up9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = keras.layers.Activation('elu')(conv9)
    conv9 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization(axis=3)(crop9)
    conv9 = keras.layers.Activation('elu')(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")
    return img


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


def generateData(batch_size, data=[]):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


def generateValidData(batch_size, data=[]):
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


if __name__ == '__main__':
    filepath = './train/bldg/'
    train_set, val_set = get_train_val()
    train_num = len(train_set)
    valid_num = len(val_set)
    print(train_num)
    print(valid_num)
    nb_epochs = 10
    batch_size = 8

    model = unet()
    model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    modelcheck = ModelCheckpoint('model_bldg_30.h5', monitor='val_acc', save_best_only=True, mode='max')
    callback = [modelcheck]
    H = model.fit_generator(generator=generateData(batch_size, train_set), steps_per_epoch=train_num//batch_size,
                            epochs=nb_epochs, verbose=1, validation_data=generateValidData(batch_size, val_set),
                            validation_steps=valid_num//batch_size, callbacks=callback, max_queue_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = 30
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plots_bldg_30.png')
