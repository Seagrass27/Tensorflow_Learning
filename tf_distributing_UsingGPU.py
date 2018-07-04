import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)
print("\n" * 5)

# 放到ipython中用runfile函数，相当于在命令行中直接运行这个脚本,试试不同规模矩阵乘法
# 在cpu和gpu上运行用时
'''runfile('D:/PythonProjects/tf_distributing_UsingGPU.py ',
        args = 'gpu 1500', wdir='D:/PythonProjects')'''

# 将device使用情况记录到日志中，当GPU不能执行某个op时，允许soft placement使得
# tensorflow自动改变device
'''with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                      log_device_placement=True)):'''