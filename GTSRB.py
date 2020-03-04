import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2
import csv
import os
from PIL import Image
import random


class Data_Set_Loader():

    x_train_set = []
    y_train_set = []

    x_test_set = []
    y_test_set = []

    X_batch = []
    Y_batch = []

    X_batch_returned = []
    Y_batch_returned = []

    X_batch_used = []

    tf_sess = None

    batch_size = 128    # Combines the number of elements into 1 batch          https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
    repeat_size = 5     # How many times each value is seen                     https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat
    shuffle = 42        # randomly selects the number of buffer_size element    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle

    def __init__(self, training_path, testing_path):

        # print("TensorFlow Version: ", tf.__version__)
        # self.tf_sess = tf.Session()
        self.load_data(training_path, testing_path)
        self.x_train_set = self.preprocess(self.x_train_set)
        self.batch_iterator(self.batch_size)

    # Loads x_train, x_test, y_train, y_test into the class variables.
    def load_data(self, training_path, testing_path):

        counter = 0
        image_counter = 0
        image_name = {}
        full_path = ''

        # for path, subdirs, files in os.walk(training_path):
        #     for name in files:
        #         if ".ppm" in name:
        #             full_path = os.path.join(path, name)
        #             image_name[full_path] = image_counter
        #             counter += 1
        #             image_counter += 1
        #             image = cv2.imread(full_path, flags=cv2.IMREAD_COLOR)
        #             self.x_train_set.append(image)

        # for path, subdirs, files in os.walk(training_path):
        #     for name in files:
        #         if '.csv' in name:
        #             with open(os.path.join(path, name)) as csv_file:
        #                 csv_reader = csv.reader(csv_file, delimiter=";")
        #                 row_count = 0
        #                 for row in csv_reader:
        #                     if(row_count > 0):
        #                         full_path_row = os.path.join(path, row[0])
        #                         self.y_train_set.append(row[7])
        #                     row_count += 1

        self.x_train_set, self.y_train_set = self.readTrafficSigns(training_path)

        print("Num Training Images:", counter)
        print("X_Train_set:", len(self.x_train_set))
        print("Y_Train_Set:", len(self.y_train_set))

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

        self.Y_batch = self.y_train_set
        self.X_batch = self.x_train_set

        print("\nNum Training Images:", counter)
        print("X_Test_set:", len(self.x_test_set))
        print("Y_Test_Set:", len(self.y_test_set))

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

        num_batches = 0

        images_used = num_images = 0

        random_position = -1

        while images_used < len(self.x_train_set):

            positions = []                                                                                      # Positions used in this iteration
            self.X_batch_returned = []                                                                          # Images returned per batch
            self.Y_batch_returned = []                                                                          # Classes returned per batch
            num_images = 0                                                                                      # Number of images read in each batch iteration

            for number in range(batch_size):                                                                    # For each image to be returned in the iteration

                while num_images < batch_size:                                                                  # While the number of images returned in this iteration is less than the required amount

                    if batch_size > (len(self.X_batch) - images_used):                                          # If the batch size requirement is greater than the number of images left to be returned
                        for num in range(len(self.x_train_set)):                                                # For every image in the training set 
                            if num not in self.X_batch_used:                                                    # For every imaged not used yet
                                if num_images < (len(self.X_batch) - images_used): 
                                    # print("Num images:", num_images, num)
                                    num_images += 1                                                                 
                                    positions.append(num)
                                    self.X_batch_returned.append(self.x_train_set[num])
                                    self.Y_batch_returned.append(self.y_train_set[num])
                                else:
                                    self.X_batch_used.extend(positions)
                                    num_batches += 1

                                    images_used += len(self.X_batch_returned)
                                    # print(len(self.X_batch_returned), "images were returned with", len(self.X_batch) - images_used, "images remaining")
                                    # print(len(self.Y_batch_returned), "images were returned with", len(self.X_batch) - images_used, "images remaining\n")

                                    # print(num_batches, "batches of images were removed")
                                    # print(images_used, "images were removed")
                                    return;
                    else:
                        random_position = random.randrange(len(self.x_train_set))

                        if random_position not in self.X_batch_used and random_position not in positions:
                            num_images += 1
                            positions.append(random_position)
                            self.X_batch_returned.append(self.x_train_set[random_position])
                            self.Y_batch_returned.append(self.y_train_set[random_position])

            self.X_batch_used.extend(positions)
            num_batches += 1

            images_used += len(self.X_batch_returned)
        
            # print(len(self.X_batch_returned), "images were returned with", len(self.X_batch) - images_used, "images remaining")
            # print(len(self.Y_batch_returned), "images were returned with", len(self.X_batch) - images_used, "images remaining\n")

        # x_set = tf.data.Dataset.from_tensor_slices(features)
        # y_set = tf.data.Dataset.from_tensor_slices(labels)
        # data = tf.data.Dataset.zip((x_set, y_set)).batch(2)

        # data = data.repeat(self.repeat_size)
        # data = data.shuffle(self.shuffle)
        # data = data.batch(self.batch_size)
        # iterator = tf.Data.Iterator.from_structure(data.output_types, data.output_shapes)
        # train_init = iterator.make_initializer(data)
        # self.X_batch, self.Y_batch = iterator.get_next()

    # function for reading the images
    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of images, list of corresponding labels
    def readTrafficSigns(self, rootpath):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        images = []  # images
        labels = []  # corresponding labels
        # loop over all 42 classes
        for c in range(0, 43):
            # subdirectory for class
            prefix = rootpath + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + 'GT-' + format(c, '05d') +
                          '.csv')  # annotations file
            # csv parser for annotations file
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                # the 1th column is the filename
                images.append(plt.imread(prefix + row[0]))
                labels.append(row[7])  # the 8th column is the label
            gtFile.close()

        # self.x_train_set = images
        # self.y_train_set = labels
        return images, labels


if __name__ == "__main__":

    loader = Data_Set_Loader(training_path='./Training',
                             testing_path='./Testing')

    n_train = len(loader.x_train_set)
    n_test = len(loader.x_test_set)
    x_n_iterator = len(loader.X_batch)
    y_n_iterator = len(loader.Y_batch)
    n_classes = len(set(loader.y_train_set))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of iterator X examples =", x_n_iterator)
    print("Number of iterator Y examples =", y_n_iterator)
    print("Number of classes =", n_classes)