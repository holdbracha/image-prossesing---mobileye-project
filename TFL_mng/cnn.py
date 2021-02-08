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

def get_model_from_files(path_to_json, path_to_h5):
    # load json and create model
    json_file = open(path_to_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_to_h5)
    return loaded_model



class CNN:
    def __init__(self):
        self.model = get_model_from_files('TFL_mng/model.json', 'TFL_mng/model.h5')
        self.image = None

    def get_image_81_81(self, x_points, y_points):
        list_images = []
        for i in range(len(x_points)):
            t_image = self.image[y_points[i]-40:y_points[i]+41, x_points[i]-40:x_points[i]+41]
            list_images.append(t_image)
        return list_images





    def is_trafic(self, crop_image):
        crop_image = crop_image.reshape([-1, 81, 81, 3])
        res = self.model.predict(crop_image)
        return res[0][1] > res[0][0]




    def find_tfl_by_cnn(self, image, points):
        self.image = np.array(image)
        trafic_images = self.get_image_81_81(list(points[:,1]), list(points[:,0]))
        real_points = []
        for i in range(len(trafic_images)):
            if trafic_images[i].shape == (81, 81, 3) and self.is_trafic(trafic_images[i]):
                real_points.append(points[i])
        print("number of points phase 2:    ", len(real_points))
        return real_points