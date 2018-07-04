import tensorflow as tf

# 求过(2,9),(-1,3)的直线系数
# Point 1
x1 = tf.constant(2, dtype = tf.float32)
y1 = tf.constant(9, dtype = tf.float32)
point1 = tf.stack([x1,y1])

# Point 2
x2 = tf.constant(-1, dtype = tf.float32)
y2 = tf.constant(3, dtype = tf.float32)
point2 = tf.stack([x2,y2])

# Combine the points into an array
X = tf.transpose(tf.stack([point1, point2]))
B = tf.ones((1,2), dtype = tf.float32)

parameters = tf.matmul(B, tf.matrix_inverse(X))

with tf.Session() as sess:
    print('Point1 is{}, Point2 is{}'.format(sess.run(point1),sess.run(point2)))
    print('X is {}'.format(sess.run(X)))
    A = sess.run(parameters)

b = 1/A[0][1]
a = -b * A[0][0]
print('Equation: y = {a}x + {b}'.format(a = a, b = b))

# 求过(2,1),(0,5),(-1,2)三点的圆系数
points = tf.constant([[2,1],[0,5],[-1,2]])
A = tf.constant([
        [2,1,1],
        [0,5,1],
        [-1,2,1]],dtype = tf.float32)
B = -tf.constant([[5],[25],[5]], dtype = tf.float32)
X = tf.matrix_solve(A,B) # 相当于求A的逆乘以B
with tf.Session() as sess:
    result = sess.run(X)
    D, E, F = result.flatten()
    
print('Equation: x**2 + y**2 + {D}x + {E}y + {F} = 0'.format(**locals()))

# =============================================================================
# Exercise:
# =============================================================================
# 2.The general form of an ellipse is given below. Solve for the following 
# points (five points are needed to solve this equation):
# General form of an ellipse:
# Ax2+By2+Cxy+Dx+Ey+F=0
# Observed points:
# (8,0),(4,-2*6**0.5),(-2*14**0.5,2),(-46**0.5,3),(14**0.5,5)

