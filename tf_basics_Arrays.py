import matplotlib.image as mpimg
import os

# dir_path = os.path.dirname(os.path.realpath('MarshOrchid.jpg'))
dir_path = os.getcwd()
filename = dir_path + '\\MarshOrchid.jpg'

# load the image as a numpy array
image = mpimg.imread(filename)

# the image is 5528 pixels high, 3685 pixels wide, and 3 colors “deep”
print(image.shape)

import matplotlib.pyplot as plt
plt.imshow(image)

# =============================================================================
# Geometric Manipulations
# =============================================================================
import tensorflow as tf

x = tf.Variable(image, name = 'x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm = [1, 0, 2]) # swap axes 0 and 1
    session.run(model)
    result = session.run(x)

plt.imshow(result)

# 将图片左右对调
import numpy as np
height, width, depth = image.shape
x = tf.Variable(image, name = 'x')
model  = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    # 在每一行上操作，对列进行调换，每一行都进行长度为width的调换
    x = tf.reverse_sequence(x,batch_axis = 0,seq_axis = 1, 
                            seq_lengths = [width]*height)
    result = session.run(x)

print(result.shape)
plt.imshow(result)

# =============================================================================
# Exercise
# =============================================================================

# 1) Combine the transposing code with the flip code to rotate clock wise.
x = tf.Variable(image, name = 'x')
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    x = tf.reverse_sequence(x, batch_dim = 1, seq_dim = 0, 
                            seq_lengths = [height]*width)
    x = tf.transpose(x, perm = [1, 0, 2])
    result = session.run(x)
plt.imshow(result)
# 2) Currently, the flip code (using reverse_sequence) requires width to 
# be precomputed. Look at the documentation for the tf.shape function, and 
# use it to compute the width of the x variable within the session.
x = tf.Variable(image, name = 'x')
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    height, width, depth = session.run(tf.shape(x))
    x = tf.reverse_sequence(x, batch_dim = 0, seq_dim = 1,
                            seq_lengths = [width]*height)
    result = session.run(x)
plt.imshow(result)
# 3) Perform a “flipud”, which flips the image top-to-bottom.
# 第一题中用了
# 4) 略