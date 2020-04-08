import numpy as np
import tensorflow as tf
from Data_Set_Loader import Data_Set_Loader
from datetime import datetime

class CNN_classifier():

    tf_sess = None
    dataset = None
    learning_rate = 0.001

    def __init__(self, dataset, num_epochs=10):
        self.tf_sess = tf.Session()
        self.dataset = dataset
        self.build()
        self.train(num_epochs)

    def create_convulational_layer(self, input, num_channels, num_filters, filter_sz):
        weights = tf.Variable(tf.random.truncated_normal(shape=[filter_sz, filter_sz, num_channels, num_filters]))
        biases = tf.Variable(tf.zeros([num_filters]))
        return tf.nn.conv2d(input, weights, [1, 1, 1, 1], 'VALID') + biases

    def new_fc_layer(self,
                     input,  # Previous layer
                     num_inputs,  # Num. of Inputs from prev layer
                     num_outputs,  # Num. of outputs
                     use_relu=True):  # Use Rectified Linear Unit (ReLU) ?

        # Create weights and biases
        weights = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.05))
        biases = tf.Variable(tf.zeros([num_outputs]))

        # Calculate the layer as matrix multiplication of inputs and weights, then
        # add bias values
        layer = tf.matmul(input, weights) + biases

        # use ReLU ?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def build(self):
        print("BUILDING THE MODEL ...")
        imageShape = [item for t in [img_shape] for item in t]
        self.x = tf.compat.v1.placeholder(tf.float32, [None] + imageShape)
        self.y = tf.compat.v1.placeholder(tf.int32, [None])

        # print("Input shape", img_shape)

        #First Convolutional Layer
        num_filters = 6
        num_channels = 3
        filter_sz = 5
        conv_layer_1 = self.create_convulational_layer(self.x, num_channels, num_filters, filter_sz)
        print("shape after 1st layer", conv_layer_1.shape)

        # first pooling
        pool_1 = tf.nn.max_pool2d(conv_layer_1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        print("shape after 1st pooling", pool_1.shape)

        # flattened layer
        current_shape = pool_1.get_shape()
        features = current_shape[1:4].num_elements()
        flattened_layer = tf.reshape(pool_1, [-1, features])
        print("shape after 1st pooling", flattened_layer.shape)

        # Fully connected layer
        fc_layer_1_input = 1176
        fc_layer_1_output = 500
        fc_layer_1 = self.new_fc_layer(flattened_layer, fc_layer_1_input, fc_layer_1_output, use_relu=True)
        print("Shape of After 1st FC:", fc_layer_1.shape)

        logits = self.new_fc_layer(fc_layer_1, 500, 43, use_relu=False)
        print("Shape after logits:", logits.shape)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(self.y, num_classes))
        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(tf.one_hot(self.y, num_classes), axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.prediction = tf.argmax(logits, axis=1)

    def train(self, epochs, limit = 10):
        print("Entered train")
        self.tf_sess.run(tf.compat.v1.global_variables_initializer())
        
        best = 0
        no_change = 0

        for epoch in range(epochs):
            # print(epoch)
            self.tf_sess.run(self.dataset.train_init)
            count = 0
            try:
                total = 0
                while 1:
                    bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])
                    
                    feed_dict = {
                        self.x: self.dataset.augment_images(bx),
                        self.y: by
                    }
                    
                    self.tf_sess.run(self.optimizer, feed_dict=feed_dict)
                    
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    total += acc * len(by)
                    
            except(tf.errors.OutOfRangeError):
                pass

            except(IndexError):
                pass

            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            print(f'epoch {epoch + 1}: loss = {loss:.4f} | training accuracy = {total / len(self.dataset.y_train_set):.4f}')
            
            if acc > best:
                best = acc
            else:
                # print("Best:", best, "| Acc:", acc)
                no_change += 1

            if no_change >= limit:
                print("EARLY STOPPING")
                break

        feed_dict = {
            self.x: self.dataset.x_test_set,
            self.y: self.dataset.y_test_set
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'Test accuracy = {acc:.4f}')

        


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    data = Data_Set_Loader("./Training", "./Testing/")
    # print("The length of the training images(x_train_set) is: ", len(data.x_train_set))
    # print("The length of the training labels(y_train_set) is: ", len(data.y_train_set))
    # print("The length of the testing images(x_test_set) is: ", len(data.x_test_set))
    # print("The length of the testing labels(y_test_set) is: ", len(data.y_test_set))
    epochs = 10
    img_shape = data.x_train_set[0].shape
    num_classes = len(np.unique(data.y_train_set))
    start = datetime.now()
    cnn = CNN_classifier(data, num_epochs=20)
    end = datetime.now()
    print("Time taken to train the model on " + str(epochs) + " epochs is:", str(end - start))
    cnn.tf_sess.close()