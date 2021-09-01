from Model import get_unet0
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import backend as K
from keras.backend import binary_crossentropy
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


img_rows = 112
img_cols = 112
smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")
    return img


def get_train_val(filepath, val_rate=0.25):
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


def generateData(filepath, batch_size, data=[]):
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


def generateValidData(filepath, batch_size, data=[]):
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
    filepath = 'F:/resources/reference/train_data/bldg/'
    train_set, val_set = get_train_val(filepath)
    train_num = len(train_set)
    valid_num = len(val_set)
    # print(train_num)
    # print(valid_num)

    batch_size = 32
    nb_epoch = 30

    model = get_unet0()
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelcheck = ModelCheckpoint('model_bldg.h5', monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
    H = model.fit_generator(generator=generateData(batch_size, train_set), steps_per_epoch=train_num // batch_size, epochs=nb_epoch,
                            verbose=1,
                            validation_data=generateValidData(batch_size, val_set), validation_steps=valid_num // batch_size,
                            callbacks=callable, max_q_size=1)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, nb_epoch), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, nb_epoch), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, nb_epoch), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, nb_epoch), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plots_bldg.png')
