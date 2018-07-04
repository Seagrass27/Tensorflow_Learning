from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import tensorflow as tf

# generate data and transform y into one-hot encoding
X_values, y_flat = make_blobs(n_features=2, n_samples=800, centers=3, 
                              random_state=500)
y = OneHotEncoder().fit_transform(y_flat.reshape(-1, 1)).todense()
y = np.array(y)


# Optional line: Sets a default figure size to be a bit larger.
plt.rcParams['figure.figsize'] = (24, 10)

plt.scatter(X_values[:,0], X_values[:,1], c=y_flat, alpha=0.4, s=150)

# split the data 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, y_train_flat, y_test_flat = \
train_test_split(X_values, y, y_flat)

X_test += np.random.randn(*X_test.shape) * 1.5 # add some noise
plt.plot(X_test[:,0], X_test[:,1], 'rx', markersize=20)

# create a model
batch_size = 21
n_features = X_train.shape[1]
n_classes = y_train.shape[1]
weights_shape = (n_features, n_classes)
train_size = X_train.shape[0]
x_train_batch = tf.placeholder(tf.float32)
y_train_batch = tf.placeholder(tf.float32)


weights = tf.Variable(dtype = tf.float32, 
                      initial_value = tf.random_normal(weights_shape))
bias = tf.Variable(dtype = tf.float32, 
                   initial_value = tf.random_normal((1, n_classes)))

y_pred_batch = tf.matmul(x_train_batch, weights) + bias
loss = tf.losses.softmax_cross_entropy(y_train_batch, y_pred_batch)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

x_test = tf.constant(X_test, dtype = tf.float32)
y_test_pred = tf.matmul(x_test, weights)

def batch_generator(step, batch_size, data):
    data_size = data.shape[0]
    offset = step * batch_size % data_size
    if offset > (data_size - batch_size + 1):
        batch_1 = data[offset:, :]
        batch_2 = data[:(batch_size-batch_1.shape[0]),:]
        batch = np.vstack((batch_1, batch_2))
    else:
        batch = data[offset : (offset + batch_size)]
    return batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        x_data = batch_generator(step, batch_size, X_train)
        y_data = batch_generator(step, batch_size, y_train)
        sess.run(train_op, feed_dict = {x_train_batch: x_data,
                                        y_train_batch: y_data})
    
        if (step % 100) == 0:
            print('Loss at step {} is : {}'.format(step, sess.run(loss,
                  feed_dict = {x_train_batch: x_data,y_train_batch: y_data})))
    test_predictions = sess.run(y_test_pred)
    bias_final, weights_final = sess.run([bias, weights], feed_dict =
                                         {x_train_batch: x_data,
                                        y_train_batch: y_data})
    
predicted_y_values = np.argmax(test_predictions, axis=1)

h = 1
x_min, x_max = X_values[:, 0].min() - 2 * h, X_values[:, 0].max() + 2 * h
y_min, y_max = X_values[:, 1].min() - 2 * h, X_values[:, 1].max() + 2 * h
x_0, x_1 = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01)) # 生成网格坐标
decision_points = np.c_[x_0.ravel(), x_1.ravel()] # 转化成列stack

Z = np.argmax(decision_points @ weights_final[[0,1]] + bias_final, axis=1)
Z = Z.reshape(x_0.shape) 

plt.contourf(x_0, x_1, Z, alpha=0.3)  # 带填充的等高图
plt.scatter(X_train[:,0], X_train[:,1], c=y_train_flat, alpha=0.3)    
plt.scatter(X_test[:,0], X_test[:,1], c=predicted_y_values, marker='x', s=200)     
plt.xlim(x_0.min(), x_0.max())
plt.ylim(x_1.min(), x_1.max())        
