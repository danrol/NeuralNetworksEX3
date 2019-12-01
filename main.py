import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    T = 0.1
    mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

    x = tf.compat.v1.placeholder(tf.float32, [None, 784], name="Inputs")
    t = tf.compat.v1.placeholder(tf.float32, [None, 10], name="Targets")

    w = tf.Variable(tf.zeros([784, 10]), name="Out_layer_w")
    b = tf.Variable(tf.zeros([10]), name="Out_biases")

    z = tf.matmul(x, w) + b
    y = tf.sigmoid(z/T)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
    train_step = tf.compat.v1.train.AdamOptimizer(0.5).minimize(cross_entropy)

    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()

    batch_xsx, batch_ys = mnist.train.next_batch(batch_size=100)

    # for x, y in zip(batch_xsx, batch_ys):
    #    print("x= " + str(x)," y=" + str(y))

    for _ in range(1000):
        batch_xsx, batch_ts = mnist.train.next_batch(100)
        print(sess.run([train_step, cross_entropy], feed_dict={x: batch_xsx, t:batch_ts}))
    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, t: mnist.test.labels}))


if __name__ == "__main__":
    main()

