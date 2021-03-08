import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
###load data
import glob


def remove_pc_outliers(pc, radius=0.1, pts=1):
    neigh = NearestNeighbors(radius=radius)
    neigh_dist, _ = neigh.fit(pc[:, :3]).radius_neighbors()
    idx = []
    for i in range(pc.shape[0]):
        if (len(neigh_dist[i]) > pts):
            idx.append(i)
    pc_new = pc[idx]
    return pc_new


for fname in glob.glob('/data/Carmel/datasets/new_modelnet40/tagging_test/*.txt', recursive=True):
    # fname = '/data/Carmel/datasets/semantic_kitti/only_people/velodyne/1186_1189_22_05_002619__.pts'
    if 'modelnet40_test' in fname or not 'person' in fname: continue
    point_cloud = pd.read_csv(fname, sep=',').to_numpy()
    # if point_cloud.shape[0] == 10000: continue
    point_cloud = remove_pc_outliers(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    ##estimate normals
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(20)
    # bbox = pcd.get_axis_aligned_bounding_box()
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    ##create mesh
    # first
    # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=14, width=0, scale=1.1, linear_fit=False)[0]
    # p_mesh_crop = poisson_mesh.crop(bbox)

    # second
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.abs(np.mean(distances))
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))
    dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()

    ##sample from mesh
    pcd = dec_mesh.sample_points_uniformly(number_of_points=10000)
    xyz = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    resulted_pc = np.concatenate((xyz, normals), axis=1)
    new_name = fname
    np.savetxt(fname, resulted_pc, delimiter=',')
    # np.savetxt(fname, point_cloud, delimiter=',')
