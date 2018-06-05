import tensorflow as tf
import numpy as np

b = tf.Variable(tf.zeros((100,)))
W = tf.Variable(tf.random_uniform((784, 100), -1, 1))

x = tf.placeholder(tf.float32, (100, 784))
h = tf.nn.relu(tf.matmul(x, W) + b)

prediction = tf.nn.softmax(h)
label = tf.placeholder(tf.float32, (None, 10))

cross_entropy = tf.reduce_min(-tf.reduce_sum(label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


graph = tf.get_default_graph()
graph.get_operations()


sess = tf.Session()
sess.run(tf.global_variables_initializer())

#res = sess.run(h, {x: np.random.random([100, 784])})

for i in range(1000):
    batch_x, batch_label = data.next_batch()
    sess.run(train_step, feed_dict={x: batch_x,
                                    label: batch_label})


with tf.variable_scope("foo"):
    v = tf.get_variable("v", shape=[1])

with tf.variable_scope('foo', reuse=True):
    v1 = tf.get_variable('v')

# with tf.variable_op_scope("foo",reuse=False):
#     v1 = tf.get_variable('v')
