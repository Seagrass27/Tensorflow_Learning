from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# create a function that plots out a basic shape
def plot_basic_object(points):
    """Plots a basic object, assuming its convex and not too complex"""
    tri = Delaunay(points).convex_hull
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                        triangles=tri,
                        shade=True, cmap=cm.Blues,lw=0.5)
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
# create a shape. The function below will return the eight points that make up a 
# cube. If you go back to the previous function, you will see the Delaunay line, 
# which turns these points into triangles, so that we may render them.
def create_cube(bottom_lower=(0, 0, 0), side_length=5):
    """Creates a cube starting from the given bottom-lower point 
    (lowest x, y, z values)"""
    bottom_lower = np.array(bottom_lower)
    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_length, 0],
        bottom_lower + [side_length, side_length, 0],
        bottom_lower + [side_length, 0, 0],
        bottom_lower + [0, 0, side_length],
        bottom_lower + [0, side_length, side_length],
        bottom_lower + [side_length, side_length, side_length],
        bottom_lower + [side_length, 0, side_length],
        bottom_lower,
    ])
    return points

cube_1 = create_cube(side_length=2)
plot_basic_object(cube_1)
