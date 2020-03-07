import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2
import csv
import os
from PIL import Image
import random
from readTrafficSigns import readTrafficSigns 


class Data_Set_Loader():

    x_train_set = []
    y_train_set = []

    x_test_set = []
    y_test_set = []

    X_batch = []
    Y_batch = []

    X_batch_returned = []
    Y_batch_returned = []

    epochs = 150

    batch_test_counter = 0
    all_positions = []

    # tf_sess = None

    batch_size = 256    # Combines the number of elements into 1 batch          https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
    shuffle = 42        # randomly selects the number of buffer_size element    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    repeat_size = 5     # How many times each value is seen                     https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat

    def __init__(self, training_path, testing_path):

        # print("TensorFlow Version: ", tf.__version__)
        # self.tf_sess = tf.Session()
        self.load_data(training_path, testing_path)
        self.x_train_set = self.preprocess(self.x_train_set)

        for epoch in tqdm(range(self.epochs)):
            self.batch_iterator(self.batch_size)

    # Loads x_train, x_test, y_train, y_test into the class variables.
    def load_data(self, training_path, testing_path):

        counter = 0
        image_counter = 0
        image_name = {}
        full_path = ''

        self.x_train_set, self.y_train_set = readTrafficSigns(training_path)

        # print("Num Training Images:", counter)
        # print("X_Train_set:", len(self.x_train_set))
        # print("Y_Train_Set:", len(self.y_train_set))

        counter = 0

        for path, subdirs, files in os.walk(testing_path):
            for name in files:
                if ".ppm" in name:
                    full_path = os.path.join(path, name)
                    image_name[full_path] = image_counter
                    counter += 1
                    image_counter += 1
                    image = cv2.imread(full_path, flags=cv2.IMREAD_COLOR)
                    self.x_test_set.append(image)

        for path, subdirs, files in os.walk(testing_path):
            for name in files:
                if '.csv' in name:
                    with open(os.path.join(path, name)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=";")
                        row_count = 0
                        for row in csv_reader:
                            if(row_count > 0):
                                full_path_row = os.path.join(path, row[0])
                                self.y_test_set.append(row[7])
                            row_count += 1

        # self.Y_batch = self.y_train_set
        # self.X_batch = self.x_train_set

        # print("\nNum Training Images:", counter)
        # print("X_Test_set:", len(self.x_test_set))
        # print("Y_Test_Set:", len(self.y_test_set))

    # Displays the image sent in.
    def display_one(self, a, title1="Original"):

        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # Processes each RGB pixel by dividing its value by 255 and saving it in [0, 1) format.
    def preprocess(self, features):
        return_vector = []

        for image in tqdm(features):
            image = np.float32(np.divide(image, 255))
            return_vector.append(image)

        return return_vector

    # Iterates through the batch to make repeat the signs, shuffle them, and select batches.
    def batch_iterator(self, batch_size):

        self.X_batch_returned = []          # Images returned per batch
        self.Y_batch_returned = []          # Classes returned per batch
        self.batch_test_counter += 1
        batch_positions = []
        images_created = 0
        images_repeated = 0
        repeated_total = 0

        while (images_created < batch_size):    # For each image to be returned in the iteration

            random_position = random.randrange(len(self.x_train_set))
            
            if self.repeat_size > 0 and random_position not in batch_positions:
                self.X_batch_returned.append(self.x_train_set[random_position])
                self.Y_batch_returned.append(self.y_train_set[random_position])

                images_created += 1
                self.repeat_size -= 1
                batch_positions.append(random_position)
            
            elif random_position not in self.all_positions and self.repeat_size == 0:
                self.all_positions.append(random_position)
                batch_positions.append(random_position)

                self.X_batch_returned.append(self.x_train_set[random_position])
                self.Y_batch_returned.append(self.y_train_set[random_position])

                images_created += 1
            else:
                images_repeated += 1

        # print("Num iteration:", self.batch_test_counter)
        # print("Num images added:", images_created, "| Total images", len(self.all_positions))
        # print("Total images generated:", (images_repeated + images_created), "| Num repeated:", images_repeated, " | Num repeated in total", repeated_total)
        # print()

if __name__ == "__main__":

    loader = Data_Set_Loader(training_path='./Training',
                             testing_path='./Testing')

    n_train = len(loader.x_train_set)
    n_test = len(loader.x_test_set)
    n_classes = len(set(loader.y_train_set))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of classes =", n_classes)