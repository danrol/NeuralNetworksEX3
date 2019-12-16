import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from functools import partial
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
T = 1  # TODO consider removing it

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
# TODO measure and display training time for each architecture
# TODO display number of weights in architecture


def logistic_regression_with_layer(n_input=784, n_output=10, n_hidden1=200, n_hidden2=200):
    logging.info("***********LOGISTIC REGRESSION WITH LAYER BEGIN***********")
    logging.info("Layer 1 size = " + str(n_hidden1) + " Layer 2 size = " + str(n_hidden2))
    seed = tf.compat.v1.set_random_seed(random.randint(1, 1000))
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
    h1 = tf.Variable(tf.random.uniform([n_input, n_hidden1], -1, 1, seed=seed), name="h1")
    b1 = tf.Variable(tf.random.uniform([1, n_hidden1], -1, 1, seed=seed), name="b1")
    h2 = tf.Variable(tf.random.uniform([n_hidden1, n_hidden2], -1, 1, seed=seed), name="h2")
    b2 = tf.Variable(tf.random.uniform([1, n_hidden2], -1, 1, seed=seed), name="b2")
    w = tf.Variable(tf.random.uniform([n_hidden2, n_output], -1, 1, seed=seed), name="Out_layer_w")
    b = tf.Variable(tf.random.uniform([1, n_output], -1, 1, seed=seed), name="Out_biases")

    h1_s = tf.add(tf.matmul(x, h1), b1)
    h1_s_rel = tf.nn.relu(h1_s / T)
    h2_s = tf.add(tf.matmul(h1_s_rel, h2), b2)
    h2_s_rel = tf.nn.relu(h2_s / T)
    z = tf.add(tf.matmul(h2_s_rel, w), b)

    return x, z


def logistic_regression(n_input=784, n_output=10):
    logging.info("***********LOGISTIC REGRESSION BEGIN***********")
    logging.info("input size = " + str(n_input) + " output size = " + str(n_output))
    seed = tf.compat.v1.set_random_seed(random.randint(1, 1000))
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
    w = tf.Variable(tf.random.uniform([n_input, n_output], -1, 1, seed=seed), name="Out_layer_w")
    b = tf.Variable(tf.random.uniform([n_output], -1, 1, seed=seed), name="Out_biases")
    z = tf.add(tf.matmul(x, w), b)
    return x, z


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=None, name="Const")
    return tf.Variable()


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def logistic_regression_conv_layers(x_input_size=28, y_input_size=28, n_input=784, n_output=10, num_filters1=32, num_filters2=64, drop_rate_percent=0.5, x_filter_size=5, y_filter_size=5, dimension_size=1, training_range=13000, batch_size=50):
    x = tf.compat.v1.placeholder(tf.float32, [None, x_input_size*y_input_size])
    t = tf.compat.v1.placeholder(tf.float32, [None, n_output])

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
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()
    # Train and evaluate
    mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=t, logits=y_conv))
    train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(training_range):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], t: batch[1], keep_prob: drop_rate_percent})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], t: batch[1], keep_prob: drop_rate_percent})
    print("Test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, t: mnist.test, keep_prob: drop_rate_percent}))


def build_train(session, network, n_output=10, training_range=13000, batch_size=50):
    x, z = network()
    x, z = train_network(session=session, input_x=x, output_z=z, n_output=n_output, training_range=training_range, batch_size=batch_size)
    return x, z


def train_network(session, input_x, output_z, n_output, training_range, batch_size):
    t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=output_z))
    train_step = tf.compat.v1.train.AdamOptimizer(name="Adam").minimize(cross_entropy)
    tf.compat.v1.global_variables_initializer().run()
    for _ in range(training_range):
        batch_xsx, batch_ts = mnist.train.next_batch(batch_size=batch_size)
        ts, ce = session.run([train_step, cross_entropy], feed_dict={input_x: batch_xsx, t: batch_ts})
    correct_prediction = tf.equal(tf.math.argmax(output_z, 1), tf.math.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    logging.info("Accuracy with training data = " + str(session.run(accuracy, feed_dict={input_x: batch_xsx, t: batch_ts})))
    logging.info("Accuracy with test data = " + str(session.run(accuracy, feed_dict={input_x: mnist.test.images, t: mnist.test.labels})))
    return input_x, output_z


def main():
    sess = tf.compat.v1.InteractiveSession()
    x, z = build_train(sess, network=logistic_regression_with_layer)
    #x, z = build_train(sess, network=logistic_regression)
    # predict try out
    for image in mnist.test.images:
        print(sess.run(tf.math.argmax(z, 1), {x: [image]}))
        first_image = image
        first_image = np.array(image, dtype='float')
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
    sess.close()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, filename='logger.log', filemode='w')
    logging.basicConfig(level=logging.INFO)
    main()
