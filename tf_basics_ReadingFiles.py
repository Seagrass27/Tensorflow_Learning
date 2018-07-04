# =============================================================================
# Placeholders
# The most basic method for reading data is to simply read it with standard 
# python code. Let’s take a look at a basic example of this, reading data 
# from this file of the 2016 Olympic Games medal tally.
# =============================================================================

# First, we create our graph, which takes a single line of data, and adds up the 
# total medals.
import tensorflow as tf
import os

dir_path = os.getcwd()
filename = dir_path + "\\olympics2016.csv"

features = tf.placeholder(tf.int32, shape=[3], name='features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')

printerop = tf.Print(total, [country, features, total], name='printer')

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            country_name, code, gold, silver, bronze, total = line.strip().\
            split(",")
            gold = int(gold)
            silver = int(silver)
            bronze = int(bronze)
            # Run the Print op
            total = sess.run(printerop, 
                             feed_dict={features: [gold, silver, bronze], 
                                        country:country_name})
            print(country_name, total)
    # 在Spyder中Print Op不能正常工作(有返回值，但是不会打印！)，
    # 但在IPython和Python中试过可以正常工作！
# It is generally fine to work in a manner similar to this. Create placeholders, 
# load a bit of data into memory, compute on it, and loop with new data. 
# This is, after all, what placeholders are for.
 ''' Doesn't work           
# =============================================================================
# Reading CSV files in TensorFlow
# =============================================================================
def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [0], [0], [0], [0]]
    country, code, gold, silver, bronze, total = tf.decode_csv(
            csv_row, record_defaults=record_defaults)
    features = tf.stack([gold, silver, bronze])
    return features, country
# The reader here technically takes a queue object, not a normal Python list, 
# so we need to build one before passing it to our function:
filename_queue = tf.train.string_input_producer([filename], 
                                                num_epochs=1, shuffle=False)
example, country = create_file_reader_ops(filename_queue)
# Those operations that result from that function call will later represent 
# single entries from our dataset. Running these requires a little more work 
# than normal. The reason is that the queue itself doesn’t sit on the graph 
# in the same way a normal operation does, so we need a Coordinator to manage 
# running through the queue. This co-ordinator will increment through the 
# dataset everytime example and label are evaluated, as they effectively pull 
# data from the file.
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while True:
        try:
            example_data, country_name = sess.run([example, country])
            print(example_data, country_name)
        except tf.errors.OutOfRangeError:
            break
'''