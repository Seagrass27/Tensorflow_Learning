import tensorflow as tf

# This is known as an broadcasted operation. Our primary object of reference 
# was a, which is a list of numbers, also called an array or a one-dimensional
# vector. Adding a single number (called a scalar) results in an broadcasted 
# operation, where the scalar is added to each element of the list.
a = tf.constant([3,4,5], name = 'a')
b = tf.constant(10, name = 'b')
add_op = a + b
with tf.Session() as session:
    print(session.run(add_op))

# In this case, the array was broadcasted to the shape of the matrix, 
# resulting in the array being added to each row of the matrix. Using this 
# terminology, a matrix is a list of rows.    
a = tf.constant([[1,2,3],[4,5,6]], name = 'a')
b = tf.constant([10,2,1], name = 'b')
add_op = a + b
with tf.Session() as session:
    print(session.run(add_op))

# This didn’t work, as TensorFlow attempted to broadcast across the rows. 
# It couldn’t do this, because the number of values in b (2) was not the 
# same as the number of scalars in each row (3).
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([100, 101], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))

# We can do this operation by creating a new matrix from our list instead.
# Due to the fact that the shapes match on the first dimension but not the 
# second, the broadcasting happened across columns instead of rows.
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([[100], [101]], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))
a.shape
b.shape
    
# =============================================================================
# Exercise:
# =============================================================================

# 1.Create a 3-dimensional matrix. What happens if you add a scalar, array or 
# matrix to it?
import numpy as np
sess = tf.InteractiveSession()
a = tf.constant(np.arange(24).reshape(2,3,4), name = 'a')
b = tf.constant(10, name = 'b')
c = tf.constant([8,9], name = 'c')
d = tf.constant([8,9,10], name = 'd')
e = tf.constant([8,9,10,11], name = 'e')
f = tf.constant([[1],[2],[3]], name = 'f')
g = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]], name = 'g')
h = tf.constant([[[10]]])
a.eval()
(a+b).eval()
(a+c).eval() # can't broadcast this way, note the error message
(a+d).eval() # can't broadcast this way, note the error message
(a+e).eval() # e.shape:(4,) a.shape(2,3,4); e broadcasted to shape(1,1,4),
             # then to shape(2,3,4)
(a+f).eval() # f.shape:(3,1) a.shape(2,3,4); f broadcasted to shape(1,3,1), 
             # then to shape (2,3,4)
(a+g).eval() # g.shape(3,4) a.shape(2,3,4); g broadcasted to shape(1,3,4),
             # then to shape (2,3,4)
(a+h).eval() # equivalent to a+b
sess.close()
# 总结：broadcast先在dimension个数少的array上已有shape前面补1（注意只能是前面），
# 直到dimension个数相等，再在每个dimension上补成相等的size

# 2.Use tf.shape (it’s an operation) to get a constant’s shape during operation 
# of the graph.
a = tf.constant(np.arange(6).reshape(1,2,3), name = 'a')
get_shape_op = tf.shape(a)

with tf.Session() as session:
    print(session.run(a))
    print(session.run(get_shape_op))
    

    
