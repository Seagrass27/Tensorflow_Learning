import tensorflow as tf

# the first dimention of placeholder is None, meaning we can have any number of
# rows 
x = tf.placeholder(dtype = 'float', shape = [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[1,2,3],
              [4,5,6]]
    result = session.run(y, feed_dict = {x: x_data})
    print(result)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

raw_image_data = mpimg.imread('MarshOrchid.jpg')

image = tf.placeholder(dtype = 'uint8', shape = [None, None, 3])
sliced = tf.slice(image, begin = [1000,0,0], size = [3000, -1, -1])

# 注意这里没有global_variables_initializer()，因为没有Variable
with tf.Session() as session:
    result = session.run(sliced, feed_dict = {image: raw_image_data})

plt.imshow(result)

# =============================================================================
# Exercise
# =============================================================================

# 1) Take a look at the other functions for arrays in TensorFlow at the official 
#    documentation.

# 2) Break the image apart into four “corners”, then stitch it back together 
#    again.
import math
height, width, depth = raw_image_data.shape
image = tf.placeholder(dtype = 'uint8', shape = [None, None, 3])
#--------------------------- break apart---------------------------------
lt_slice = tf.slice(image, begin = [0,0,0], size = [math.ceil(height/2),
                    math.ceil(width/2),-1])
rt_slice = tf.slice(image, begin = [0, math.ceil(width/2),
                                    0], size = [math.ceil(height/2), -1, -1])
lb_slice = tf.slice(image, begin = [math.ceil(height/2), 0, 0],
                                    size = [-1, math.ceil(width/2), -1])
rb_slice = tf.slice(image, begin = [math.ceil(height/2),math.ceil(width/2),0],
                                    size = [-1, -1, -1]) 
with tf.Session() as session:
    lt = session.run(lt_slice, feed_dict={image:raw_image_data})
    rt = session.run(rt_slice, feed_dict={image:raw_image_data})
    lb = session.run(lb_slice, feed_dict={image:raw_image_data})
    rb = session.run(rb_slice, feed_dict={image:raw_image_data})

fig = plt.figure()
ax1 = fig.add_axes([0, .1, .5, .4])
ax2 = fig.add_axes([.5, .1, .5, .4])
ax3 = fig.add_axes([0, .5, .5, .4])
ax4 = fig.add_axes([.5, .5, .5, .4])

ax1.imshow(lb)
ax2.imshow(rb)
ax3.imshow(lt)
ax4.imshow(rt)
#-----------------------stitch together------------------------------------    
top_half = tf.concat([lt_slice, rt_slice], axis = 1)
bot_half = tf.concat([lb_slice, rb_slice], axis = 1)
whole_picture = tf.concat([top_half,bot_half], axis = 0)

with tf.Session() as session:
    recovery = session.run(whole_picture, feed_dict = {image: raw_image_data})

plt.imshow(recovery)
(recovery == raw_image_data).all()

# 3) Convert the image into grayscale. One way to do this would be to take 
#    just a single colour channel and show that. Another way would be to 
#    take the average of the three channels as the gray colour.

# 方法一
image = tf.placeholder(dtype = 'uint8')
gray_picture = tf.slice(image,[0,0,0],[-1,-1,1])
with tf.Session() as session:
    result = session.run(gray_picture, feed_dict = {image: raw_image_data})
    
plt.imshow(result[:,:,0], cmap = 'gray') # 需要把第三维去掉才能显示

# 方法二
image = tf.placeholder(dtype = 'uint8')
gray_picture = tf.reduce_mean(image, axis = 2)
with tf.Session() as session:
    result = session.run(gray_picture, feed_dict = {image: raw_image_data})
print(result.shape)
plt.imshow(result, cmap = 'gray')
