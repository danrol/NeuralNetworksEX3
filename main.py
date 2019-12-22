import logging
# from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
import os
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)


# TODO display number of weights in architecture


def main():
    # logging.basicConfig(level=logging.INFO, filename='logger.log', filemode='w')
    logging.basicConfig(level=logging.INFO)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #  this command forces to use CPU instead of GPU
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    networks = []
    # the following for loops set the network type and other parameters
    for batch_size in [10, 50, 100]:
        for train_range in [100, 13000]:
            net = Network(session=sess, network_type=logistic_regression, batch_size=batch_size, num_iterations=train_range)
            networks.append(net)

            net = Network(session=sess, network_type=logistic_regression_with_layer, batch_size=batch_size, num_iterations=train_range)
            networks.append(net)

            for dropout in [0.5]:
                net = Network(session=sess, network_type=logistic_regression_conv_layers, batch_size=batch_size, num_iterations=train_range, keep_probability_value=dropout)
                networks.append(net)

    # begin training networks - run performs buildTrain, predict, score and visualize
    for network in networks:
        network.run()
        logging.info(network.scores_str())
    sess.close()


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=None, name="Const")
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Network:
    # scores for data will be kept in the following order:
    # fscore, accuracy, recall, precision = ['train', 'validation', 'test']
    class DataType:
        Names = ["TRAIN", "VALIDATION", "TEST"]
        TRAIN = 0
        VALIDATION = 1
        TEST = 2

    def __init__(self, session, network_type, batch_size=100, num_iterations=13000, keep_probability_value=0.5):
        """
        :type session: tensorflow session object
        :type network_type: network creator function pointer
        """
        self.fscore = [0, 0, 0]
        self.accuracy = [0, 0, 0]
        self.recall = [0, 0, 0]
        self.precision = [0, 0, 0]
        self.keep_probability_value = keep_probability_value
        self.x_placeholder = None
        self.z_variables = None
        self.keep_prob_placeholder = None
        self.target_placeholder = None
        self.session = session
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.net_type = network_type
        self.net = None
        self.train_time = timedelta(seconds=0)

    def run(self):
        self.build_train()
        self.test_all_scores()
        self.visualize()

    def build_train(self):
        self.net = self.net_type()  # create network

        self.x_placeholder = self.net[0]
        self.z_variables = self.net[1]
        self.keep_prob_placeholder = self.net[2]
        self.target_placeholder = self.net[3]

        self.train_network()
        logging.info(str(self))

    def score(self, y_values, t_values, data_type_enum, printlog=False):
        y_val, t_val = np.argmax(y_values, 1), np.argmax(t_values, 1)

        self.accuracy[data_type_enum] = accuracy_score(t_val, y_val)

        self.fscore[data_type_enum] = f1_score(t_val, y_val, average="macro")

        self.precision[data_type_enum] = precision_score(t_val, y_val, average="macro", zero_division=0)

        self.recall[data_type_enum] = recall_score(t_val, y_val, average="macro", zero_division=0)
        if printlog:
            logging.info(self.scores_str(data_type_enum))

    def test_all_scores(self):
        logging.debug("test_all_scores begin")

        y_values, t_values = self.predict(data=mnist.train, use_batch=True)
        self.score(y_values, t_values, Network.DataType.TRAIN)

        y_values, t_values = self.predict(data=mnist.validation, use_batch=True)
        self.score(y_values, t_values, Network.DataType.VALIDATION)

        y_values, t_values = self.predict(data=mnist.test, use_batch=True)
        self.score(y_values, t_values, Network.DataType.TEST)
        logging.debug("test_all_scores end")

    def predict(self, data, use_batch=False):
        if use_batch:
            batch_x, batch_t = data.next_batch(self.batch_size)
        else:
            batch_x, batch_t = data.images, data.labels  # use whole database

        if self.keep_prob_placeholder is None:  # check if conv net or not
            y = self.session.run(self.net[1], feed_dict={self.x_placeholder: batch_x})
        else:
            y = self.session.run(self.net[1], feed_dict={self.x_placeholder: batch_x, self.keep_prob_placeholder: 1})
        return y, batch_t

    def scores_str(self, data_type_enum=None):
        s = ""
        if data_type_enum is not None:
            if data_type_enum == Network.DataType.TRAIN:
                s = s + "\n"
            s = s + " - " + Network.DataType.Names[data_type_enum] + ": "
            s = s + "Network Scores\n\tAccuracy: " + str(self.accuracy[data_type_enum]) + " Fscore: " + str(self.fscore[data_type_enum]) + "\n\tPrecision: " + str(self.precision[
                                                                                                                                                                       data_type_enum]) \
                + " recall: " + str(self.recall[data_type_enum]) + "\n"
        else:
            s = self.scores_str(Network.DataType.TRAIN)
            s = s + (self.scores_str(Network.DataType.VALIDATION))
            s = s + (self.scores_str(Network.DataType.TEST))
        return s

    def __str__(self):
        s = "Batch_size=" + str(self.batch_size) + " Num training iterations=" + str(self.num_iterations) + " dropout_rate=" + str(self.keep_probability_value)
        s = s + "\n\t\tTraining time is : " + str(self.train_time)
        return s

    def visualize(self):
        pass  # TODO

    def train_network(self):
        iteration_number_for_target_accuracy = None
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_placeholder, logits=self.z_variables))
        train_step = tf.compat.v1.train.AdamOptimizer(name="Adam").minimize(cross_entropy)
        tf.compat.v1.global_variables_initializer().run()
        t1 = timer()

        for _ in range(self.num_iterations):
            batch_xsx, batch_ts = mnist.train.next_batch(batch_size=self.batch_size)
            if self.keep_prob_placeholder is None:
                ts, ce = self.session.run([train_step, cross_entropy], feed_dict={self.x_placeholder: batch_xsx, self.target_placeholder: batch_ts})
            else:
                ts, ce = self.session.run([train_step, cross_entropy], feed_dict={self.x_placeholder: batch_xsx, self.target_placeholder: batch_ts, self.keep_prob_placeholder: self.keep_probability_value})

            if iteration_number_for_target_accuracy is None:
                y_values, t_values = self.predict(data=mnist.validation, use_batch=True)
                self.score(y_values, t_values, Network.DataType.VALIDATION)
                if self.accuracy[Network.DataType.VALIDATION] >= 0.99:
                    iteration_number_for_target_accuracy = _
                    t3 = timer()
                    break
        if iteration_number_for_target_accuracy is not None:
            logging.info("Reached 99% accuracy within " + str(iteration_number_for_target_accuracy) + " iterations and " + str(timedelta(seconds=t3 - t1)))
        t2 = timer()
        self.train_time = timedelta(seconds=t2 - t1)
        logging.info("Training time is : " + str(self.train_time))


def logistic_regression_with_layer(n_input=784, n_output=10, n_hidden1=200, n_hidden2=200):
    logging.info("***********LOGISTIC REGRESSION WITH LAYER***********")
    logging.info("Layer 1 size = " + str(n_hidden1) + " Layer 2 size = " + str(n_hidden2))
    seed = tf.compat.v1.set_random_seed(random.randint(1, 1000))
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
    t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")
    h1 = tf.Variable(tf.random.uniform([n_input, n_hidden1], -1, 1, seed=seed), name="h1")
    b1 = tf.Variable(tf.random.uniform([1, n_hidden1], -1, 1, seed=seed), name="b1")
    h2 = tf.Variable(tf.random.uniform([n_hidden1, n_hidden2], -1, 1, seed=seed), name="h2")
    b2 = tf.Variable(tf.random.uniform([1, n_hidden2], -1, 1, seed=seed), name="b2")
    w = tf.Variable(tf.random.uniform([n_hidden2, n_output], -1, 1, seed=seed), name="Out_layer_w")
    b = tf.Variable(tf.random.uniform([1, n_output], -1, 1, seed=seed), name="Out_biases")

    h1_s = tf.add(tf.matmul(x, h1), b1)
    h1_s_rel = tf.nn.relu(h1_s)
    h2_s = tf.add(tf.matmul(h1_s_rel, h2), b2)
    h2_s_rel = tf.nn.relu(h2_s)
    z = tf.add(tf.matmul(h2_s_rel, w), b)

    return [x, z, None, t, h1_s, h1_s_rel, h2_s, h2_s_rel]


def logistic_regression(n_input=784, n_output=10):
    logging.info("***********LOGISTIC REGRESSION***********")
    logging.info("input size = " + str(n_input) + " output size = " + str(n_output))
    seed = tf.compat.v1.set_random_seed(random.randint(1, 1000))
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
    t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")
    w = tf.Variable(tf.random.uniform([n_input, n_output], -1, 1, seed=seed), name="Out_layer_w")
    b = tf.Variable(tf.random.uniform([n_output], -1, 1, seed=seed), name="Out_biases")
    z = tf.add(tf.matmul(x, w), b)
    return [x, z, None, t]


def logistic_regression_conv_layers(x_input_size=28, y_input_size=28, n_input=784, n_output=10, num_filters1=32, num_filters2=64, drop_rate_percent=0.5, x_filter_size=5,
                                    y_filter_size=5, dimension_size=1, training_range=13000, batch_size=50):
    logging.info("***********LOGISTIC REGRESSION WITH CONVOLUTION***********")
    x = tf.compat.v1.placeholder(tf.float32, [None, x_input_size * y_input_size])
    t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")
    w_conv1 = weight_variable([x_filter_size, y_filter_size, dimension_size, num_filters1])  # 32 filters of 5x5x1 (x axis, y axis, dimension (greyscale is 1))
    b_conv1 = bias_variable([num_filters1])

    w_conv2 = weight_variable([x_filter_size, y_filter_size, num_filters1, num_filters2])
    b_conv2 = bias_variable([num_filters2])

    x_image = tf.reshape(x, [-1, x_input_size, y_input_size, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pooling_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pooling_2x2(h_conv2)

    # Fully connected layer 1024
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.compat.v1.placeholder(tf.float32)
    rate = 1 - keep_prob
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=rate)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    tf.compat.v1.global_variables_initializer().run()
    return [x, y_conv, keep_prob, t, h_conv1, h_pool1, h_conv2, h_pool2, h_pool2_flat, h_fc1]


if __name__ == "__main__":
    main()
