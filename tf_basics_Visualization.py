import tensorflow as tf

a = tf.add(1, 2,)
b = tf.multiply(a, 3)
c = tf.add(4, 5,)
d = tf.multiply(c, 6,)
e = tf.multiply(4, 5,)
f = tf.div(c, 6,)
g = tf.add(b, d)
h = tf.multiply(g, f)

with tf.Session() as sess:
	print(sess.run(h))
# Now we add a SummaryWriter to the end of our code, this will create a 
# folder in your given directory, Which will contain the information for 
# TensorBoard to build the graph.
with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()

# 打开Anaconda Prompt，
# tensorboard --logdir=d://PythonProjects//output
# 打开网页127.0.0.1:6006就可以看到这个graph

# =============================================================================
# Adding names
# =============================================================================
a = tf.add(1, 2, name="Add_these_numbers")
b = tf.multiply(a, 3)
c = tf.add(4, 5, name="And_These_ones")
d = tf.multiply(c, 6, name="Multiply_these_numbers")
e = tf.multiply(4, 5, name="B_add")
f = tf.div(c, 6, name="B_mul")
g = tf.add(b, d)
h = tf.multiply(g, f)

with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()
    
# =============================================================================
# Creating scopes
# If we give the graph a name by typing with tf.name_scope("MyOperationGroup"): 
# and give the graph a scope like this with tf.name_scope("Scope_A"):. when 
# you re-run your TensorBoard you will see something very different. 
# The graph is now much more easier to read, and you can see that it all 
# comes under the graph header, In this case that is MyOperationGroup, and 
# then you have your scopes A and B, Which have there operations within 
# them.
# =============================================================================
#Here we are defining the name of the graph, scopes A, B and C.
with tf.name_scope("MyOperationGroup"):
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="Add_these_numbers")
        b = tf.multiply(a, 3)
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="And_These_ones")
        d = tf.multiply(c, 6, name="Multiply_these_numbers")

with tf.name_scope("Scope_C"):
    e = tf.multiply(4, 5, name="B_add")
    f = tf.div(c, 6, name="B_mul")
g = tf.add(b, d)
h = tf.multiply(g, f)

with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()


