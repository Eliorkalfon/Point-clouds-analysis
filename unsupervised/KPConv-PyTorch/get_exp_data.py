import numpy as np
import pandas as pd
import glob
import shutil
import open3d as o3d


def normalize(points):
    # points = points[:, :3]
    centroid = np.mean(points[:, :3], axis=0)
    points[:, :3] -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points[:, :3]) ** 2, axis=-1)))
    points[:, :3] /= furthest_distance
    return points


def z_permutation(pc, max_displacement=0.05):
    random_vector = np.random.uniform(-max_displacement, max_displacement, len(pc))
    new_pc = np.c_[(pc[:, 0], pc[:, 1], pc[:, 2] + random_vector, pc[:, 3],pc[:, -3],pc[:, -2],pc[:, -1])]
    return new_pc


if __name__ == '__main__':
    org = '/data/Carmel/users/elior/regions_of_interest/CL_90/cc_cl90/'
    dest = '/data/Carmel/datasets/cc_based_modelnet40/cl90/'

    f_list = []
    counter = 0
    for file in glob.glob(org + '*.txt', recursive=True):
        pc = pd.read_csv(file, sep=' ').to_numpy().reshape(-1, 9)
        pc = pc[np.where(pc[:,2]>pc[:,2].min()+0.1)]
        if pc.shape[0]<50:
            continue
        counter+=1

        normalized_pc = normalize(pc)
        new_pc = np.c_[(
            normalized_pc[:, 0], normalized_pc[:, 2], normalized_pc[:, 1], normalized_pc[:, 3], normalized_pc[:, 5],
            normalized_pc[:, 4])]
        f_name = str(counter)+'_'+file.split('/')[-1].replace('pts', 'txt')
        f_list.append(f_name)
        dest_norm = dest + f_name
        np.savetxt(dest_norm, new_pc, delimiter=',')
    with open(dest + 'modelnet40_test.txt', 'w') as f:
        for item in f_list:
            f.write("%s\n" % item)
