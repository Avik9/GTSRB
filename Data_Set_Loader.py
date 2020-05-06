import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2
import csv
import os
from PIL import Image
from random import randint


class Data_Set_Loader():

    x_train_set = []
    y_train_set = []

    x_test_set = []
    y_test_set = []

    x_batch = []
    y_batch = []

    epochs = 10
    shape = None
    num_classes = None
    session = None

    batch_test_counter = 0
    all_positions = []
    train_init = None

    batch_size = 512    # Combines the number of elements into 1 batch          https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
    shuffle = 512       # randomly selects the number of buffer_size element    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle

    def __init__(self, training_path, testing_path):

        self.session = tf.compat.v1.Session()

        # print("TensorFlow Version: ", tf.__version__)
        self.load_data("./Training", "./Testing/")
        self.batch_iterator(self.batch_size)

    # Loads x_train, x_test, y_train, y_test into the class variables.
    def load_data(self, training_path, testing_path):

        labels = np.array([])
        images = []
        count_files = 0
        count_dirs = 0

        for c in range(43):
            prefix = training_path + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + '/' + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)
            for row in gtReader:
                im = cv2.imread(prefix + row[0])
                im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                im = np.divide(im, 255.0)
                images.append(im)
                labels = np.append(labels, row[7])

            gtFile.close()


        self.x_train_set = np.asarray(images)
        self.y_train_set = labels
        # print("Images:",type(images))

        # print("\nNum Testing Images:")
        # print("X_Test_set:", len(self.x_train_set))
        # print("Y_Test_Set:", len(self.y_train_set))

        labels = []
        images = []
        count_files = 0
        # prefix = 'Testing/'
        gtFile = open(testing_path + 'GT-final_test.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        for row in gtReader:
            im = cv2.imread(testing_path + row[0])
            im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            
            # Processes each RGB pixel by dividing its value by 255 and saving it in [0, 1) format.
            im = np.divide(im, 255.0)
            images.append(im)
            labels.append(row[7])
        gtFile.close()

        self.x_test_set = images
        self.y_test_set = labels

        # print("\nNum Training Images:")
        # print("X_Test_set:", len(self.x_test_set))
        # print("Y_Test_Set:", len(self.y_test_set))

    # Displays the image sent in.
    def display_one(self, a, title1="Original"):

        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # Iterates through the batch to make repeat the signs, shuffle them, and select batches.
    def batch_iterator(self, batch_size):
        self.x_batch = []          # Images returned per batch
        self.y_batch = []          # Classes returned per batch

        data_xy = tf.data.Dataset.from_tensor_slices((self.x_train_set, self.y_train_set))
        data_xy = data_xy.shuffle(len(self.y_train_set), reshuffle_each_iteration=True).batch(self.batch_size)

        iterator = tf.compat.v1.data.Iterator.from_structure(data_xy.output_types, data_xy.output_shapes)
        self.train_init = iterator.make_initializer(data_xy)
        self.x_batch, self.y_batch = iterator.get_next()

    def augment_images(self, images):
        to_augment = randint(0, 1)
        # print("Predicted", to_augment)
        if to_augment == 1:
            augmented_images = [] 
            for pos in range(self.batch_size):
                img = images[pos]
                choice = randint(0, 2)

                if choice == 0:
                    img = img;
                    # print("Image", pos, "was not augmented")

                if choice == 1:
                    img = self.translateImage(img)
                    # print("Image", pos, "was translated")

                if choice == 2:
                    img = self.gaussianNoise(img)
                    # print("Image", pos, "had gaussian noise added")
                
                img = np.divide(img, 255.0)
                # img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                augmented_images.append(img) #, axis=0)
                # print("Image shape:", img.shape)
        
            return np.array(augmented_images)
        return np.array(images)

    def translateImage(self, image, height=32, width=32, max_x_trans=5, max_y_trans=5):
        translate_x = max_x_trans * np.random.uniform() - max_x_trans / 2
        translate_y = max_y_trans * np.random.uniform() - max_y_trans / 2
        translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        trans = cv2.warpAffine(image, translation_mat, (height, width))

        return trans

    def gaussianNoise(self, image, ksize=(3, 3), border=0):
        return cv2.GaussianBlur(image, ksize, border)

if __name__ == "__main__":

    loader = Data_Set_Loader(training_path='./Training',
                             testing_path='./Testing')

    n_train = len(loader.x_train_set)
    n_test = len(loader.x_test_set)
    n_classes = len(set(loader.y_train_set))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of classes =", n_classes)

    imageToTest = loader.x_train_set[345]
    loader.display_one(imageToTest, "Original")

    imageToTest = loader.translateImage(imageToTest)
    loader.display_one(imageToTest, "Translated")

    imageToTest = loader.x_train_set[525]
    loader.display_one(imageToTest, "Original")

    imageToTest = loader.gaussianNoise(imageToTest)
    loader.display_one(imageToTest, "Added Gaussian Noise")

    img_shape = loader.x_train_set[0].shape
    num_classes = len(np.unique(loader.y_train_set))

    loader.session.close()