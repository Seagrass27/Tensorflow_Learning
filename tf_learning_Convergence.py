import tensorflow as tf
import numpy as np

# perform updates in a for loop
x = tf.Variable(0, name = 'x')
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    for i in range(5):
        x += 1
        print(sess.run(x))

# perform updates in a while loop
x = tf.Variable(0., name = 'x')
threshold = tf.constant(5., name = 'threshold')
model = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(model)
    while sess.run(tf.less(x, threshold)):
        x = tf.add(x, 1)
    print(sess.run(x))
    
# =============================================================================
# Gradient Descent
# =============================================================================

# a simple optimization
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable([4., 2.])
y_pred = tf.multiply(x, w[0]) + w[1]
err = tf.square(y - y_pred)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(err)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_value = np.random.rand()
    y_value = x_value * 5 + 3
    for i in range(10000):
        sess.run(train_op, feed_dict = {x: x_value, y: y_value})
    print(sess.run(w))
    
# Other Optimisation:
    
# GradientDescentOptimizer
# AdagradOptimizer
# MomentumOptimizer
# AdamOptimizer
# FtrlOptimizer
# RMSPropOptimizer
    
# Plot the error:
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable([3., 2.])
y_pred = tf.multiply(x, w[0]) + w[1]
err = tf.square(y - y_pred)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(err)
errs = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_train = np.random.rand()
    y_train = x_train * 5 + 3
    for i in range(1000):
        _, error = sess.run([train_op, err], feed_dict = {x:x_train,y:y_train})
        errs.append(error)
    w = sess.run(w)
import matplotlib.pyplot as plt
plt.plot([np.mean(errs[i : i+50]) for i in range(len(errs) - 49)])

# =============================================================================
# Exercise:
# =============================================================================
# 3.Our example trains on just a single example at a time, which is inefficient.
# Extend it to learn using a number (say, 50) of training samples at a time.
x = tf.placeholder(tf.float32, shape = (10, 1))
y = tf.placeholder(tf.float32, shape = (10, 1))
w = tf.Variable([3., 3.])
y_pred = tf.multiply(x, w[0]) + w[1]
err = tf.reduce_mean(tf.square(y - y_pred))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(err)
errs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_batch = np.random.rand(10, 1)
    y_batch = x_batch * 5 + 4
    for step in range(1000):
        _, error = sess.run([train_op, err], 
                            feed_dict = {x: x_batch, y: y_batch})
        errs.append(error)
    print(sess.run(w))
plt.plot([np.mean(errs[i : i+50]) for i in range(len(errs) - 49)])