# Generate some samples
import tensorflow as tf
import numpy as np


def create_samples(n_clusters, n_samples_per_cluster, n_features, 
                   embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, 
                                   seed=seed, name="cluster_{}".format(i))
        current_centroid = (
        (np.random.random((1, n_features)) * embiggen_factor) - 
        (embiggen_factor/2)
        )
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples


n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, 
                                    n_samples_per_cluster, 
                                    n_features, embiggen_factor, seed)

model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)

# creat a plotting function

import matplotlib.pyplot as plt
  
def plot_clusters(all_samples, centroids, n_samples_per_cluster):
     
     #Plot out the different clusters
     #Choose a different color for each cluster
     colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
     for i, centroid in enumerate(centroids):
         # Grab just the samples for the given cluster and plot them out 
         # with a new colour
         samples = (all_samples
         [i*n_samples_per_cluster:(i+1)*n_samples_per_cluster])
         plt.scatter(samples[:,0], samples[:,1], c=colour[i])
         #Also plot centroid
         plt.plot(centroid[0], centroid[1], markersize=35, marker="x", 
                  color='k', mew=10)
         plt.plot(centroid[0], centroid[1], markersize=30, marker="x", 
                  color='m', mew=5)

plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

# Initialization
def choose_random_centroids(samples, n_clusters):
    # 从samples中随机选3个作为起始centroids
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0,]
    size = [n_clusters]
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)#按indices选元素
    return initial_centroids 

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, 
                                    n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)

model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(initial_centroids)

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)

# Updating centroids
def assign_to_nearest(samples, centroids):
    # Finds the nearest centroid for each sample
    expanded_vectors = tf.expand_dims(samples, 0) #给samples增加一个维度
    expanded_centroids = tf.expand_dims(centroids, 1) #给centroids增加一个维度
    distances = tf.reduce_sum( tf.square( #利用broadcast求距离，得到3*1500数组
               tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0) #得到(1500,)数组，也就是每个点最近的centroid
    nearest_indices = mins
    return nearest_indices
def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    # 按0， 1， 2进行分割
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat(
            [tf.expand_dims(tf.reduce_mean(partition, 0), 0) 
            for partition in partitions], 0)
    return new_centroids

data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, 
                                         n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

# a single iteration of k-means
model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    updated_centroid_value = session.run(updated_centroids)
    print(updated_centroid_value)

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)
