import cv2
import random
import os
import numpy as np


img_w = 256
img_h = 256

image_sets = ['1.png', '2.png', '3.png', '4.png']


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    return xb, yb


def creat_dataset(image_num=50000, mode='original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in range(len(image_sets)):
        count = 0
        src_img = cv2.imread('./data/src/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('./data/water_label/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  # single channel
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            cv2.imwrite(('./train/water/src/%d.png' % g_count), src_roi)
            cv2.imwrite(('./train/water/label/%d.png' % g_count), label_roi)
            count += 1
            g_count += 1


if __name__ == '__main__':
    creat_dataset(image_num=80000, mode='augment')