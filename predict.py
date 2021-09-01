
import numpy as np
import os
from keras.models import load_model
from skimage import io

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

palette = {0: (0, 0, 0),        # Undefined (black)
           1: (0, 255, 0),      # Trees (green)
           2: (255, 0, 0),      # Buildings (red)
           3: (0, 0, 255),      # Water (blue)
           4: (255, 255, 0)}    # Roads (yellow)
mask1_pool = ['trees.png', 'bldg.png',
              'water.png', 'road.png']

def convert_to_color(arr_2d, palette=palette):
    # Numeric labels to RGB-color encoding
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def predict(pre_model, img, name, init_size=112, final_size=80):
    model = pre_model
    image = img
    H, W = image.shape[0], image.shape[1]

    shift = int((init_size - final_size) / 2)

    if H % final_size == 0:
        num_h_tiles = int(H / final_size)
    else:
        num_h_tiles = int(H / final_size) + 1

    if W % final_size == 0:
        num_w_tiles = int(W / final_size)
    else:
        num_w_tiles = int(W / final_size) + 1

    rounded_height = num_h_tiles * final_size
    rounded_width = num_w_tiles * final_size

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((padded_height, padded_width, 3))

    padded[shift:shift + H, shift: shift + W, :] = image

    # add mirror reflections to the padded areas
    up = padded[shift:2 * shift, shift:-shift, :][::-1, ...]
    padded[:shift, shift:-shift, :] = up

    lag = padded.shape[0] - H - shift
    bottom = padded[H + shift - lag:shift + H, shift:-shift, :][::-1, ...]
    padded[H + shift:, shift:-shift, :] = bottom

    left = padded[:, shift:2 * shift, :][:, ::-1, :]
    padded[:, :shift, :] = left

    lag = padded.shape[1] - W - shift
    right = padded[:, W + shift - lag:shift + W, :][:, ::-1, :]

    padded[:, W + shift:, :] = right

    h_start = range(0, padded_height, final_size)[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size)[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[h:h + init_size, w:w + init_size, :]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((rounded_height, rounded_width, 1))

    for j_h, h in enumerate(h_start):
        for j_w, w in enumerate(w_start):
            i = len(w_start) * j_h + j_w
            predicted_mask[h: h + final_size, w: w + final_size, :] = prediction[i]

    io.imsave('./cache/{}.png'.format(name), (predicted_mask[:H, :W, 0] * 255).astype(np.uint8))


def merge_and_visuliza():
    image = io.imread('./cache/{}'.format(mask1_pool[0]))
    H, W = image.shape[0], image.shape[1]
    final_mask = np.zeros((H, W), np.uint8)

    img = np.zeros((4, H, W), np.uint8)
    for idx, name in enumerate(mask1_pool):
        img[idx] = io.imread('./cache/' + name, as_gray=True)

    trees_index = 0
    bldg_index = 1
    water_index = 2
    road_index = 3
    # let's remove everything from water
    water = (img[water_index] == 1)
    for i in [0, 1, 3]:
        img[i][water] = 0

    # let's remove bldg from trees
    trees = (img[trees_index] == 1)
    img[bldg_index][trees] = 0

    # let's remove everything from road
    road = (img[road_index] == 1)
    for i in [0, 1, 2]:
        img[i][road] = 0

    for index in range(img.shape[0]):
        label_value = index + 1  # coressponding labels value
        for i in range(H):  # priority:building>water>road>vegetation
            for j in range(W):
                if img[index, i, j] == 1:
                    if label_value == 2:
                        final_mask[i, j] = label_value
                    elif label_value == 3 and final_mask[i, j] != 2:
                        final_mask[i, j] = label_value
                    elif label_value == 4 and final_mask[i, j] != 2 and final_mask[i, j] != 3:
                        final_mask[i, j] = label_value
                    elif label_value == 1 and final_mask[i, j] == 0:
                        final_mask[i, j] = label_value

    final_mask = convert_to_color(final_mask)
    io.imsave('p.png', final_mask)


def pre_processing(image_id):
    image = np.array(io.imread('./data/src/{}'.format(image_id)))

    model_trees = load_model('./cache/model_trees_30.h5')
    model_bldg = load_model('./cache/model_bldg_20.h5')
    model_water = load_model('./cache/model_water_30.h5')
    model_road = load_model('./cache/model_road_20.h5')

    predict(model_trees, image, 'trees')
    predict(model_bldg, image, 'bldg')
    predict(model_water, image, 'water')
    predict(model_road, image, 'road')

    merge_and_visuliza()


pre_processing('3.png')

# import cv2
# import random
# import numpy as np
# import os
# import argparse
# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
# # from sklearn.preprocessing import LabelEncoder
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# TEST_SET = ['1.png', '2.png', '3.png']
#
# image_size = 256
#
# classes = [0., 1., 2., 3., 4.]
#
# # labelencoder = LabelEncoder()
# # labelencoder.fit(classes)
#
# def predict():
#     # load the trained convolutional neural network
#     print("[INFO] loading network...")
#     model = load_model('model_bldg_30.h5')
#     stride = 128
#     for n in range(len(TEST_SET)):
#         path = TEST_SET[n]
#         # load the image
#         image = cv2.imread('./data/src/' + path)
#         h, w, _ = image.shape
#         padding_h = (h // stride + 1) * stride
#         padding_w = (w // stride + 1) * stride
#         padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
#         padding_img[0:h, 0:w, :] = image[:, :, :]
#         # padding_img = padding_img.astype("float") / 255.0
#         padding_img = img_to_array(padding_img)
#         print('src:', padding_img.shape)
#         mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
#         for i in range(padding_h // stride):
#             for j in range(padding_w // stride):
#                 crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size]
#                 ch, cw, _ = crop.shape
#                 if ch != 256 or cw != 256:
#                     print
#                     'invalid size!'
#                     continue
#
#                 crop = np.expand_dims(crop, axis=0)
#                 print(crop.shape)
#                 pred = model.predict(crop, verbose=2)
#                 # print (np.unique(pred))
#                 pred = pred.reshape((256, 256)).astype(np.uint8)
#                 # print 'pred:',pred.shape
#                 mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]
#
#         cv2.imwrite('pre' + str(n + 1) + '.png', convert_to_color(np.round(mask_whole[0:h, 0:w])))
#
#
# if __name__ == '__main__':
#     predict()









