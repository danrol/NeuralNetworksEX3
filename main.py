import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

WITH_LAYER = True


def main():
    T = 1
    n_input = 784  # 28x28
    n_output = 10
    n_hidden1 = 200
    n_hidden2 = 200
    mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

    if WITH_LAYER:
        x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
        t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")

        h1 = tf.Variable(tf.zeros([n_input, n_hidden1]), name="h1")
        b1 = tf.Variable(tf.zeros([1, n_hidden1]), name="b1")
        h2 = tf.Variable(tf.zeros([n_hidden1, n_hidden2]), name="h2")
        b2 = tf.Variable(tf.zeros([1, n_hidden2]), name="b2")

        w = tf.Variable(tf.zeros([n_hidden2, n_output]), name="Out_layer_w")
        b = tf.Variable(tf.zeros([1, n_output]), name="Out_biases")

        h1_s = tf.matmul(x, h1) + b1
        h1_s_zig = tf.nn.relu(h1_s / T)
        h2_s = tf.matmul(h1_s_zig, h2) + b2
        h2_s_zig = tf.nn.relu(h2_s / T)
        z = tf.matmul(h2_s_zig, w) + b
        y = tf.nn.relu(z / T)

    else:
        x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="Inputs")
        t = tf.compat.v1.placeholder(tf.float32, [None, n_output], name="Targets")

        w = tf.Variable(tf.zeros([n_input, n_output]), name="Out_layer_w")
        b = tf.Variable(tf.zeros([n_output]), name="Out_biases")

        z = tf.matmul(x, w) + b
        y = tf.sigmoid(z/T)


    # TODO change sigmoid to RelU

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
    train_step = tf.compat.v1.train.AdamOptimizer(0.5).minimize(cross_entropy)

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

