# Point Clouds Analysis
Algorithmic methods for finding anomalies in point clouds
We wish to find anomalies in a noisy background in the humans search and rescue field, using unsupervised classifiers.
Assuming the scanned scene is large and contains many points we split our data into smaller windows, depends on the objects of interest's dimensions.

# Chamfer using PCU library 
<img src="https://images.slideplayer.com/11/3315099/slides/slide_5.jpg" width="300" height="200">
Statistic measurment of the difference between point clouds

# Modified Hausdorff
M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object matching.
    In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
https://ieeexplore.ieee.org/document/576361

# Radial - NN Based Disimilarities Detector
Find points in one point cloud without neighboors in the other point cloud ,for given radius (default radius = 0.2).

# Mean and Std

Divied both point clouds into smaller windows and find each window's mean and std for each window and calculate every vector's mean.
The absolute value of the difference between the resulted statistic for each point cloud is given.


# Intensity Based Detector For 4d Point Clouds (x,y,z,intensity)


