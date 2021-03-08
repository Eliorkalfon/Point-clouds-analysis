#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:11:30 2021

@author: carmis
"""

import glob
import os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import random


def rescale(points):
    points = points[:,:3]
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    return points


def z_permutation(pc, max_displacement=0.05):
    random_vector = np.random.uniform(-max_displacement, max_displacement, len(pc))
    new_pc = np.c_[(pc[:,0], pc[:,1], pc[:,2]+random_vector, pc[:,3])]
    return new_pc


def resample(pc, num_pts):
    # random sampling
    # Finding the farthest points in a point cloud (reduce high density points)
    point_set = pc[:,:3]
    dist = pairwise_distances(point_set, metric='euclidean', n_jobs=25)
    if len(point_set) > num_pts:
        indices, distances = getGreedyPerm(dist)
        point_set = pc[indices[:num_pts], :]
        # mean = np.mean(point_set, axis=0)
        # point_set = point_set - np.expand_dims(mean, 0)  # center
        # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        # point_set = point_set / dist  # scale
    return point_set


def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    
    Parameters
    ----------
    D : ndarray (N, N) 
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list) 
        (permutation (N-length array of indices), 
        lambdas (N-length array of insertion radii))
    """
    
    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)


def remove_pc_outliers(pc, radius=1, pts=1):
    """

    :param pc: np point cloud
    :param radius: scalar in meters, default - 0.5
    :param pts: minimum amount of point neighbors, under it the point is removed
    :return: new filtered point cloud
    """
    neigh = NearestNeighbors(radius=radius)
    neigh_dist, _ = neigh.fit(pc[:, :3]).radius_neighbors()
    idx = []
    for i in range(pc.shape[0]):
        if len(neigh_dist[i]) > pts:
            idx.append(i)
    pc_new = pc[idx]
    return pc_new

i = 0
if __name__== 'main':
    data_dir = 'pc_dir' # don't forget to change
    max_displacement = 0.05
    kitty_data = False
    min_points_to_sample_from = 500
    n_samples = 5
    min_object_points = 80
    max_object_points = 500
    class_name = 'plant' # change it to a class whice relevant to you
    os.makedirs(f'{data_dir}/generated_pcs', exist_ok=True)
    
    for pcl in glob.glob(data_dir + '/*.txt', recursive=True):
        if kitty_data:
            pc = np.loadtxt(pcl, delimiter=' ')
            permute_pc = z_permutation(pc, max_displacement)
            new_pc = np.c_[(permute_pc[:,0], permute_pc[:,2], permute_pc[:,1])]
        else:
            pc = np.loadtxt(pcl, delimiter=',')
        
        np.savetxt(f'{data_dir}/generated_pcs/{class_name}_{i}orig.txt', pc, delimiter=',', fmt='%s')
        random_n_points = np.random.randint(min_object_points, max_object_points, size=n_samples)
        if pc.shape[0] > min_points_to_sample_from:       
            for pts in random_n_points:
                if pc.shape[0] > pts:
                    sampled_pts = resample(pc, pts)
                    sampled_pts = remove_pc_outliers(sampled_pts, radius=0.18, pts=1)
                    normalized_pc = rescale(pc)
        np.savetxt(f'{data_dir}/generated_pcs/{class_name}_gen{i}.txt', pc, delimiter=',', fmt='%s')
        i += 1   