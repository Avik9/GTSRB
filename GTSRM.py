import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from os import listdir, walk
from os.path import isfile, join


class Data_Set_Loader():

    x_train_set = []
    y_train_set = []
    x_test_set = []
    y_test_set = []

    def __init__(self, training_path, testing_path):
            
        counter = 0

        for path, subdirs, files in walk(training_path):
            for name in files:
                if ".ppm" in name:
                    counter += 1
                    image = cv2.imread(join(path, name), flags=cv2.IMREAD_COLOR)
                    self.x_train_set.append(image)

        print("Num Training Files:", counter)

        counter = 0

        for path, subdirs, files in walk(testing_path):
            for name in files:
                if ".ppm" in name:
                    counter += 1
                    image = cv2.imread(join(path, name), flags=cv2.IMREAD_COLOR)
                    self.x_test_set.append(image)

        print("Num Testing Files:", counter)


if __name__ == "__main__":

    training_path = 'Training'

    loader = Data_Set_Loader('Training', 'Testing')

    # print(loader.train_set[32])
    # print(len(loader.train_set))
