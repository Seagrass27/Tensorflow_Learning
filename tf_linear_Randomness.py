# create a basic helper function that simply runs a single TensorFlow variable.

import tensorflow as tf

def run_variable(variable):
    model = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(model)
        return sess.run(variable)

# Let’s start with a basic distribution, the uniform distribution.
my_distribution = tf.random_uniform((6, 4), seed=42) # 返回的是Tensor，其实不用
                                                     # initialize
uniform = run_variable(my_distribution)

# To visualise this, we can use a histogram:
from matplotlib import pyplot as plt
plt.hist(uniform.flatten())

# Normal distribution.
distribution = tf.random_normal((600,4), seed = 42)
normal = run_variable(distribution)
plt.hist(normal.flatten())

# We can specify the mean and standard deviation
distribution = tf.random_normal((10000,), seed = 42, mean = 170, stddev = 15)
normal = run_variable(distribution)
plt.hist(normal,bins = 20)

# We can use TensorFlow to create these histograms as well!
# The histogram_fixed_width function takes a list of values (like our random 
# values), the range, and the number of bins to compute. It then counts how
# many values are within the range of each bin, and returns the result as an 
# array.
import numpy as np
bins = tf.histogram_fixed_width(normal, (normal.min(), normal.max()), nbins=20)
histogram_bins = run_variable(bins)
x_values = np.linspace(normal.min(), normal.max(), len(histogram_bins))
plt.bar(x_values, histogram_bins)
# That’s correct, but doesn’t look right. The histogram values are there, but 
# the widths are unusually narrow (our bins are represented by single values 
# only). Let’s fix that:
bar_width = (normal.max() - normal.min()) / len(histogram_bins)
plt.bar(x_values, histogram_bins, width=bar_width)

# =============================================================================
# Exercise:
# =============================================================================
# 1.Use a Uniform distribution to model a single dice-roll. Plot the result to 
# ensure it is consistent with your expectations
distribution = tf.random_uniform([1000000,], 1, 7,dtype = 'int32',seed = 42)
dice_roll = run_variable(distribution)
plt.hist(dice_roll)
# 2.Replace the last code block of this lesson with pure TensorFlow calls in a 
# single graph. In other words, use TensorFlow concepts to replace the .min(), 
# .max(), and len calls. Only the plotting should be done without TensorFlow!
distribution = tf.random_normal((10000,), seed = 42, mean = 170, stddev = 15)
min_val = tf.reduce_min(distribution)
max_val = tf.reduce_max(distribution)
bins = tf.histogram_fixed_width(distribution, (min_val, max_val), nbins=20)
n_bins = tf.shape(bins)[0]
x_values = tf.linspace(min_val, max_val, n_bins)
bar_width = (max_val-min_val)/tf.to_float(n_bins)
#bar_width = tf.div(tf.subtract(max_val,min_val), tf.to_float(n_bins))

plt.bar(run_variable(x_values),run_variable(bins),
        width = run_variable(bar_width))