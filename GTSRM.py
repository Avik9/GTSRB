import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import cv2
from os import listdir, walk
from os.path import isfile, join

class Data_Set_Loader():

    train_set = None
    test_set = None

    def __init__(self):
        pass

if __name__ == "__main__":

    training_path = 'Training'

    loader = Data_Set_Loader()

    # loader.train_set = pickle.load(open(training_path, 'rb'))
    # loader.x_train, loader.y_train = loader.train_set['features'], loader.train['labels']

    # onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]

    counter = 0

    for path, subdirs, files in walk(training_path):
        for name in files:
            print (join(path, name))
            if ".ppm" in name:
                counter += 1

    print("Num Files:", counter)

    # print(onlyfiles)