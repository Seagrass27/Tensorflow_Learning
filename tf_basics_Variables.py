# =============================================================================
# This is quite a bit of boilerplate, but it works like this:
# 
# 1.Import the tensorflow module and call it tf
# 2.Create a constant value called x, and give it the numerical value 35
# 3.Create a Variable called y, and define it as being the equation x + 5
# 4.Initialize the variables with tf.global_variables_initializer() (we will 
#   go into more detail on this)
# 5.Create a session for computing the values
# 6.Run the model created in 4
# 7.Run just the variable y and print out its current value

# The step 4 above is where some magic happens. In this step, a graph is 
# created of the dependencies between the variables. In this case, the 
# variable y depends on the variable x, and that value is transformed by 
# adding 5 to it. Keep in mind that this value isn’t computed until step 7, 
# as up until then, only equations and relations are computed.
# =============================================================================
import tensorflow as tf
 
x = tf.constant(35, name = 'x')
y = tf.Variable(x+5, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))

# =============================================================================
# Exercise
# =============================================================================
    
# 1.Constants can also be arrays. Predict what this code will do, then 
# run it to confirm:
x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))

# 2.Generate a NumPy array of 10,000 random numbers (called x) and create a 
#  Variable storing the equation
#  y=5x2−3x+15               (1)
#  You can generate the NumPy array using the following code:
import numpy as np
data = np.random.randint(1000, size=10000)
# This data variable can then be used in place of the list from question 1 
# above. 
x = tf.constant(data,name = 'x')
y = tf.Variable(5*x**2+15, name = 'y')    
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print(session.run(y))
    
# 3.You can also update variables in loops, which we will use later for machine 
# learning. Take a look at this code, and predict what it will do (then run 
# it to check):
x = tf.Variable(0, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        x = x + 1
        print(session.run(x))
        
# 4.Using the code from (2) and (3) above, create a program that computes 
# the “rolling” average of the following line of code: 
# np.random.randint(1000). In other words, keep looping, and in each 
# loop, call np.random.randint(1000) once in that loop, and store the 
# current average in a Variable that keeps updating each loop. 
c_sum = tf.constant(0, name = 'c_sum')
c_avg = tf.Variable(0, name = 'c_avg')
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    for i in range(10):
        x = np.random.randint(1000)
        print('The %d element generated is %d' %(i, x))
        c_sum += x
        print('The current sum is {}'.format(session.run(c_sum)))
        c_avg = c_sum/(i+1)
        print('The current average is {}\n'.format(session.run(c_avg)))