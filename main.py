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


def logistic_regression_with_layer(n_input=784, n_output=10, n_hidden1=200, n_hidden2=200):
    logging.info("***********LOGISTIC REGRESSION WITH LAYER BEGIN***********")
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
    logging.info("***********LOGISTIC REGRESSION BEGIN***********")
    logging.info("input size = " + str(n_input) + " output size = " + str(n_output))
    seed = tf.compat.v1.set_random_seed(random.randint(1, 1000))
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
    t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")
    w = tf.Variable(tf.random.uniform([n_input, n_output], -1, 1, seed=seed), name="Out_layer_w")
    b = tf.Variable(tf.random.uniform([n_output], -1, 1, seed=seed), name="Out_biases")
    z = tf.add(tf.matmul(x, w), b)
    return [x, z, None, t]


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


def logistic_regression_conv_layers(x_input_size=28, y_input_size=28, n_input=784, n_output=10, num_filters1=32, num_filters2=64, drop_rate_percent=0.5, x_filter_size=5,
                                    y_filter_size=5, dimension_size=1, training_range=13000, batch_size=50):
    logging.info("***********LOGISTIC REGRESSION WITH CONVOLUTION BEGIN***********")
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


def build_train(session, network, n_output=10, training_range=13000, batch_size=50, keep_prob_value=0.5):
    net = network()
    train_network(session=session, net=net, training_range=training_range, batch_size=batch_size,
                  keep_prob_val=keep_prob_value)
    return net


def train_network(session, net, training_range, batch_size, keep_prob_val=1.0):
    iteration_number = None
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=net[3], logits=net[1]))
    train_step = tf.compat.v1.train.AdamOptimizer(name="Adam").minimize(cross_entropy)
    tf.compat.v1.global_variables_initializer().run()
    t1 = timer()

    for _ in range(training_range):
        batch_xsx, batch_ts = mnist.train.next_batch(batch_size=batch_size)
        batch_xsx_val, batch_ts_val = mnist.validation.next_batch(batch_size=batch_size)
        if net[2] is None:
            ts, ce = session.run([train_step, cross_entropy], feed_dict={net[0]: batch_xsx, net[3]: batch_ts})
        else:
            ts, ce = session.run([train_step, cross_entropy], feed_dict={net[0]: batch_xsx, net[3]: batch_ts, net[2]: keep_prob_val})

        if iteration_number is None:
            scores = score(session=session, net=net, data=mnist.validation)
            accuracy = scores[0]
            if accuracy >= 0.99:
                iteration_number = _
                t3 = timer()
                break
    if iteration_number is not None:
        logging.info("Reached 99% accuracy within " + str(iteration_number) + " iterations and " + str(timedelta(seconds=t3 - t1)))
    t2 = timer()
    logging.info("Training time is : " + str(timedelta(seconds=t2 - t1)))


def score(session, net, data, printlog=False):
    values = predict(session=session, net=net, data=data)
    y_val, t_val = np.argmax(values[0], 1), np.argmax(values[1], 1)

    accuracy = accuracy_score(t_val, y_val)

    fscore = f1_score(t_val, y_val, average="macro")

    precision = precision_score(t_val, y_val, average="macro", zero_division=0)

    recall = recall_score(t_val, y_val, average="macro")
    if printlog:
        logging.info("Network Scores\nAccuracy: " + str(accuracy) + " Fscore: " + str(fscore) + "\nPrecision: " + str(precision) + " recall: " + str(recall))
    return [accuracy, fscore, precision, recall]


def predict(session, net, data):
    batch_x, batch_t = data.next_batch(10000)
    # TODO change next batch to take the whole data
    if net[2] is None:  # check if conv net or not
        y = session.run(net[1], feed_dict={net[0]: batch_x})
    else:
        y = session.run(net[1], feed_dict={net[0]: batch_x, net[2]: 1})
    return [y, batch_t]


def visualize():
    pass  # TODO


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #  this command forces to use CPU instead of GPU
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)
    networks = []
    scores = []
    for batch_size in [50, 100]:
        for train_range in [13000]:

            net = build_train(sess, network=logistic_regression, training_range=train_range, batch_size=batch_size)
            s = score(session=sess, net=net, data=mnist.test, printlog=True)
            # TODO send train, test and validation
            networks.append(net)
            scores.append(s)

            net = build_train(sess, network=logistic_regression_with_layer, training_range=train_range, batch_size=batch_size)
            s = score(session=sess, net=net, data=mnist.test, printlog=True)
            # TODO send train, test and validation
            networks.append(net)
            scores.append(s)

            for dropout in [0.5]:
                net = build_train(sess, network=logistic_regression_conv_layers, training_range=train_range, batch_size=batch_size,
                                  keep_prob_value=dropout)
                s = score(session=sess, net=net, data=mnist.test, printlog=True)
                # TODO send train, test and validation
                networks.append(net)
                scores.append(s)

    sess.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='logger.log', filemode='w')
    # logging.basicConfig(level=logging.INFO)
    main()
