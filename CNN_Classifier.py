"""
1 Layer CNN Classifier for GTRSB

Code retrieved from Ajay Sarjoo (GitHub: @asarj)
"""

import numpy as np
import tensorflow as tf
import os
import warnings

class CNN():

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128

    def __init__(self, dataset, num_epochs=100):
        self.tf_sess = tf.Session()
        self.dataset = dataset
        self.build_model()
        self.train_model(num_epochs)

    def build_model(self, learning_rate=0.001):
        print("Building model...")
        self.x = self.dataset.x_train_set
        self.y = self.dataset.y_train_set

        # First layer
        c1_channels = 3
        c1_filters = 6
        c1 = self.conv_layer(input=self.x, input_channels=c1_channels, filters=c1_filters, filter_size=5)

        # Pooling
        pool1 = self.pool(layer=c1, ksize=[1,2,2,1], strides=[1,2,2,1])

        # Flattened layer
        flattened = self.flatten_layer(layer=pool1)

        # First Fully Connected Layer
        fc1_input = 936
        fc1_output = 500
        fc1 = self.fc_layer(input=flattened, inputs=fc1_input, outputs=fc1_output, relu=True)

        # Logits
        l_inp = 500
        l_out = 43
        logits = self.fc_layer(input=fc1, inputs=l_inp, outputs=l_out, relu=False)

        y_to_one_hot = tf.one_hot(self.y, self.dataset.num_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_to_one_hot)

        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_to_one_hot, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.prediction = tf.argmax(logits, axis=1)

    def train_model(self, epochs):
        # print("Training model...")
        self.tf_sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            self.tf_sess.run(self.dataset.train_init)
            try:
                total = 0
                while 1:
                    bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])

                    feed_dict = {
                        self.x: bx,
                        self.y: by
                    }
                    self.tf_sess.run(self.optimizer, feed_dict=feed_dict)
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    total += acc * len(by)

            except(tf.errors.OutOfRangeError):
                pass


            feed_dict = {
                self.x: self.dataset.x_valid,
                self.y: self.dataset.y_valid
            }

            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            # print(f'epoch {epoch + 1}: loss = {loss:.4f}, training accuracy = {total / len(self.dataset.y_train):.4f}, validation accuracy = {acc:.4f}')

        feed_dict = {
            self.x: self.dataset.x_test,
            self.y: self.dataset.y_test
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        # print(f'test accuracy = {acc:.4f}')

    def create_weights(self, shape, stddev=0.05):
        return tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=stddev))

    def create_biases(self, size):
        return tf.Variable(tf.zeros([size]))

    def conv_layer(self, input, input_channels, filters, filter_size):
        weights = self.create_weights(shape=[filter_size, filter_size, input_channels, filters])
        biases = self.create_biases(filters)

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='VALID')
        layer += biases

        layer = tf.nn.relu(layer)

        return layer

    def pool(self, layer, ksize, strides, padding='VALID'):
        return tf.nn.max_pool(layer, ksize=ksize, strides=strides, padding=padding)

    def flatten_layer(self, layer):
        shape = layer.get_shape()
        features = shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, features])

        return layer
 
    def fc_layer(self, input, inputs, outputs, relu=True):
        weights = self.create_weights(shape=[inputs, outputs])
        biases = self.create_biases(outputs)

        layer = tf.matmul(input, weights)
        layer += biases

        if relu:
            layer = tf.nn.relu(layer)

        return layer