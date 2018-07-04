# =============================================================================
# Game of Life is an interesting computer science simulation
# =============================================================================
#1. If the cell is alive, but has one or zero neighbours, it “dies” through 
#   under-population.
#2. If the cell is alive and has two or three neighbours, it stays alive.
#3. If the cell has more than three neighbours it dies through over-population.
#4. Any dead cell with three neighbours regenerates.
import tensorflow as tf
from matplotlib import pyplot as plt

# Create a board
shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

with tf.Session() as session:
    X = session.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')

# Upadate the board
import numpy as np
from scipy.signal import convolve2d

def update_board(X):
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2)) #element-wise
    return X
'''
# py_func allows us to take a python function and turn it into a node in 
# TensorFlow.
board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])

with tf.Session() as session:
    initial_board_values = session.run(initial_board)
    X = session.run(board_update, feed_dict={board: initial_board_values})[0]
'''   
