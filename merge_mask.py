import numpy as np
import cv2
import csv
from tqdm import tqdm
from visuliza import convert_to_color

mask1_pool = ['trees.png', 'bldg.png',
              'water.png', 'road.png']

# after mask combind
img_sets = ['pre1.png', 'pre2.png', 'pre3.png']


def combind_all_mask():

    image = cv2.imread('./cache/{}'.format(mask1_pool[0]))
    H, W = image.shape[0], image.shape[1]
    final_mask = np.zeros((H, W), np.uint8)

    img = np.zeros((4, H, W), np.uint8)
    for idx, name in enumerate(mask1_pool):
        img[idx] = cv2.imread('./cache/' + name, 0)

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
        for i in tqdm(range(H)):  # priority:building>water>road>vegetation
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
    cv2.imwrite('./final_result/p.png', final_mask)

combind_all_mask()


