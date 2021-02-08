import numpy as np, matplotlib.pyplot as plt
from os.path import join
from keras.models import model_from_json
import seaborn as sbn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation,MaxPooling2D,BatchNormalization,Activation, Conv2D
from tensorflow.keras.models import load_model
import os
import json
import glob
import argparse
from scipy import ndimage, misc
import numpy as np
from scipy import signal as sg
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import maximum_filter1d
from functools import reduce
from scipy.spatial import distance
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt







def remove_close_values(values):
    res = []
    for i in range(len(values)):
        flag = False
        for j in range(i + 1, len(values)):
            if abs(distance.euclidean(values[i], values[j])) < 40:
                flag = True
                break
        if not flag:
            res.append(values[i])
    return res


def crop_red_or_green_kernel(path, box, red_or_green=0):
    img = Image.open(path)
    kernel = np.array(img.crop(box)).astype(float)[:, :, red_or_green] / 255
    kernel -= np.average(kernel)
    plt.imshow(kernel)
    return kernel


def get_potential_tfl_points(conv_img):
    max_image_conv = ndimage.maximum_filter(conv_img, size=40)
    threshold = (np.array(conv_img).max() - np.array(conv_img).min()) * 0.3
    non_traffic_values = np.argwhere(conv_img < threshold)
    for point in non_traffic_values:
        conv_img[point[0], point[1]] = 0
    diff = np.array(max_image_conv - conv_img)
    result = np.argwhere(diff == 0)
    return result


def get_five_kernels():
    dict_of_kernels = {}
    green_kernel = crop_red_or_green_kernel('TFL_mng/kernel/aachen_000015_000019_leftImg8bit.png', (845, 298, 855, 307),
                                            1)  # 20x15
    big_green_kernel = crop_red_or_green_kernel('TFL_mng/kernel/aachen_000046_000019_leftImg8bit.png', (1888, 244, 1918, 266),
                                                1)  # 55x45
    very_small_green_kernel = crop_red_or_green_kernel('TFL_mng/kernel/aachen_000012_000019_leftImg8bit.png',
                                                       (1020, 338, 1031, 348), 1)
    red_kernel = crop_red_or_green_kernel('TFL_mng/kernel/aachen_000046_000019_leftImg8bit.png',
                                          box=(1000, 355, 1016, 369))  # 29x29
    big_red_kernel = crop_red_or_green_kernel('TFL_mng/kernel/bremen_000121_000019_leftImg8bit.png',
                                              box=(1403, 239, 1425, 256))  # 40x40
    small_red_kernel = crop_red_or_green_kernel('TFL_mng/kernel/berlin_000042_000019_leftImg8bit.png',
                                                (1062, 325, 1079, 337))  # 17x14
    very_small_red_kernel = crop_red_or_green_kernel('TFL_mng/kernel/aachen_000013_000019_leftImg8bit.png',
                                                     (1395, 361, 1405, 370))  # 10x10
    dict_of_kernels['green'] = [green_kernel, big_green_kernel, very_small_green_kernel]
    dict_of_kernels['red'] = [red_kernel, big_red_kernel, small_red_kernel, very_small_red_kernel]
    return dict_of_kernels


def get_five_arrays_after_convolution(img, dict_of_kernels):
    list_of_ndarrays_after_convolution = []
    for kernel in dict_of_kernels['green']:
        test_array = np.array(img).astype(float)[:, :, 1] / 255
        list_of_ndarrays_after_convolution.append(ndimage.filters.convolve(test_array,
                                                                           kernel, mode='mirror', cval=0))

    for kernel in dict_of_kernels['red']:
        test_array = np.array(img).astype(float)[:, :, 0] / 255
        list_of_ndarrays_after_convolution.append(ndimage.filters.convolve(test_array,
                                                                           kernel, mode='mirror', cval=0))

    return list_of_ndarrays_after_convolution


def get_neighbors(point):

    neighbors = []
    for x in range(point[0] - 2, point[0] + 3):
        for y in range(point[1] - 2, point[1] + 3):
            if 0 <= x < 1024 and 0 <= y < 2048:
                neighbors.append((x, y))
    return neighbors


def get_average_of_neighbors(point, red_or_green, c_image):
    image_array = np.array(c_image)
    neighbors = get_neighbors(point)
    sum_ = 0
    for neighbor in neighbors:
        sum_ += image_array[neighbor][red_or_green]
    return sum_ / 25


def split_by_red_green_blue(potential_points, c_image):
    red_light_points = []
    green_light_points = []
    blue_light_points = []
    image_array = np.array(c_image)
    for point in potential_points:
        # if image_array[point[0], point[1], 0] > image_array[point[0], point[1], 1]:
        if get_average_of_neighbors(point, 0, image_array) - 10 > get_average_of_neighbors(point, 1, image_array):
            red_light_points.append(point)
        elif get_average_of_neighbors(point, 1, image_array) - 10 > get_average_of_neighbors(point, 0, image_array):
            green_light_points.append(point)
        else:
            blue_light_points.append(point)
    return np.array(red_light_points), np.array(green_light_points), np.array(blue_light_points)


def find_tfl_lights(c_image, **kwargs):
    dict_of_kernels = get_five_kernels()
    conv_images = get_five_arrays_after_convolution(c_image, dict_of_kernels)
    potential_points = []

    for conv_img in conv_images:
        potential_points.extend(get_potential_tfl_points(np.array(conv_img)).tolist())

    # print("len before:    ", len(potential_points))
    potential_points = remove_close_values(potential_points)
    # print("len before:    ", len(potential_points))
    potential_points = np.array(potential_points)
    print("number of points phase 1:    ", len(potential_points))
    return potential_points





    # *******************************************************
    trafic_images = get_image_81_81(np.array(c_image), list(potential_points[:,1]), list(potential_points[:,0]))
    real_points = []
    for i in range(len(trafic_images)):
        if trafic_images[i].shape == (81, 81, 3):
            real_points.append(potential_points[i])
    print("number of points:    ", len(real_points))
    # --------------------------------------------------------------------------------
    return real_points
    # --------------------------------------------------------------------------------

    red, green, blue = split_by_red_green_blue(np.array(real_points), c_image)
    red_x = [] if red.size == 0 else red[:, 1]
    red_y = [] if red.size == 0 else red[:, 0]
    green_x = [] if green.size == 0 else green[:, 1]
    green_y = [] if green.size == 0 else green[:, 0]
    blue_x = [] if blue.size == 0 else blue[:, 1]
    blue_y = [] if blue.size == 0 else blue[:, 0]
    return red_x, red_y, green_x, green_y, blue_x, blue_y


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(im, json_path=None, fig_num=None):
    """
    Run the attention code
    """

    image = np.array(im)
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)
    # --------------------------------------------------------------------------------
    return find_tfl_lights(im, some_threshold=42)
    # --------------------------------------------------------------------------------
    red_x, red_y, green_x, green_y, blue_x, blue_y = find_tfl_lights(im, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=3)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=3)
    plt.plot(blue_x, blue_y, 'ro', color='b', markersize=3)
    return [red_x + green_x + blue_x, red_y + green_y + blue_y]
    # plt.imshow(image)
    # plt.show()
    


def main():
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    default_base = r'/content/drive/MyDrive/testing'

    flist = glob.glob(os.path.join(default_base, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)



