import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
WITH_LAYER = True


def main():
    T = 1
    n_input = 784  # 28x28
    n_output = 10
    n_hidden1 = 200
    n_hidden2 = 200
    mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
    seed_num = random.randint(1, 1000)
    if WITH_LAYER:
        x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
        t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")

        h1 = tf.Variable(tf.random.uniform([n_input, n_hidden1], -1, 1, seed=seed_num), name="h1")
        b1 = tf.Variable(tf.random.uniform([1, n_hidden1], -1, 1, seed=seed_num), name="b1")
        h2 = tf.Variable(tf.random.uniform([n_hidden1, n_hidden2], -1, 1, seed=seed_num), name="h2")
        b2 = tf.Variable(tf.random.uniform([1, n_hidden2], -1, 1, seed=seed_num), name="b2")

        w = tf.Variable(tf.random.uniform([n_hidden2, n_output], -1, 1, seed=seed_num), name="Out_layer_w")
        b = tf.Variable(tf.random.uniform([1, n_output], -1, 1, seed=seed_num), name="Out_biases")

        h1_s = tf.add(tf.matmul(x, h1), b1)
        h1_s_rel = tf.nn.relu(h1_s / T)
        h2_s = tf.add(tf.matmul(h1_s_rel, h2), b2)
        h2_s_rel = tf.nn.relu(h2_s / T)
        z = tf.add(tf.matmul(h2_s_rel, w), b)
        y = tf.nn.relu(z / T)

    else:
        x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
        t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")

        w = tf.Variable(tf.random.uniform([n_input, n_output], -1, 1, seed=0), name="Out_layer_w")
        b = tf.Variable(tf.random.uniform([n_output], -1, 1, seed=0), name="Out_biases")

        z = tf.add(tf.matmul(x, w), b)
        y = tf.nn.relu(z/T)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
    train_step = tf.compat.v1.train.AdamOptimizer().minimize(cross_entropy)

    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()

    # batch_xsx, batch_ys = mnist.train.next_batch(batch_size=100)

    # for x, y in zip(batch_xsx, batch_ys):
    #    print("x= " + str(x)," y=" + str(y))

    for _ in range(13000):
        batch_xsx, batch_ts = mnist.train.next_batch(100)
        print(sess.run([train_step, cross_entropy], feed_dict={x: batch_xsx, t: batch_ts}))
    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, t: mnist.test.labels}))


if __name__ == "__main__":
    main()

