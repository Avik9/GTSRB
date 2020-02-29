import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import csv
import matplotlib.pyplot as plt
from os import listdir, walk
from os.path import isfile, join


class Data_Set_Loader():

    x_train_set = []
    y_train_set = []
    x_test_set = []
    y_test_set = []

    def __init__(self, training_path, testing_path):

        self.load_data(training_path, testing_path)
        self.display_one(self.x_train_set[1100])

    # Loads x_train, x_test, y_train, y_test into the class variables

    def load_data(self, training_path, testing_path):

        counter = 0
        image_counter = 0
        image_name = {}
        full_path = ''

        for path, subdirs, files in walk(training_path):
            for name in files:
                if ".ppm" in name:
                    full_path = join(path, name)
                    image_name[full_path] = image_counter
                    counter += 1
                    image_counter += 1
                    image = cv2.imread(full_path, flags=cv2.IMREAD_COLOR)
                    self.x_train_set.append(image)

        for path, subdirs, files in walk(training_path):
            for name in files:
                if '.csv' in name:
                    with open(join(path, name)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=";")
                        row_count = 0
                        for row in csv_reader:
                            if(row_count > 0):
                                full_path_row = join(path, row[0])
                                self.y_train_set.append(
                                    image_name[full_path_row])
                            row_count += 1

        # print("Num Training Images:", counter)
        # print("X_Train_set:", len(self.x_train_set))
        # print("Y_Train_Set:", len(self.y_train_set))

        counter = 0

        for path, subdirs, files in walk(testing_path):
            for name in files:
                if ".ppm" in name:
                    full_path = join(path, name)
                    image_name[full_path] = image_counter
                    counter += 1
                    image_counter += 1
                    image = cv2.imread(full_path, flags=cv2.IMREAD_COLOR)
                    self.x_test_set.append(image)

        for path, subdirs, files in walk(testing_path):
            for name in files:
                if '.csv' in name:
                    with open(join(path, name)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=";")
                        row_count = 0
                        for row in csv_reader:
                            if(row_count > 0):
                                full_path_row = join(path, row[0])
                                self.y_test_set.append(
                                    image_name[full_path_row])
                            row_count += 1

        # print("\nNum Training Images:", counter)
        # print("X_Test_set:", len(self.x_test_set))
        # print("Y_Test_Set:", len(self.y_test_set))

    # Displays the image sent in

    def display_one(self, a, title1="Original"):

        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":

    training_path = 'Training'

    loader = Data_Set_Loader('Training', 'Testing')
