import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
import multiprocessing
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import point_cloud_utils as pcu
from scipy.spatial.distance import cdist
from tools.vis_detector import *

min_num_points = 10
intensity_thd = np.inf
leaves_height = 1.8
overlap = 0.5


def multi_proc(pc0_file_paths, pc1_file_paths, model_name, use_smaller_windows=False):
    """
    pc0 and pc1 filepaths ordered by matching windows
    :param pc1_filespaths: pc0 file list
    :param p0_filespaths: pc1 file list
    :return: dictionary - features
    """
    use_smaller_windows = [use_smaller_windows for i in range(len(pc0_file_paths))]
    n_workers = 32

    # =========================================== #
    # Without using multi-processing
    # for i, k in zip(pc0_file_paths, pc1_file_paths):
    #     res = model_name(i, k, False)
    pool = multiprocessing.Pool(processes=n_workers)
    results = pool.starmap(model_name, zip(pc0_file_paths, pc1_file_paths, use_smaller_windows))
    pool.close()
    pool.join()
    results = np.asarray(results)
    if 'mean' in str(model_name):
        results = {"z_means": results[:, 0], "z_stds": results[:, 1]}
    elif 'hausdorff' in str(model_name):
        # results = (results - results.min())/(results.max() - results.min())
        results = {"hausdorff": results}
    elif 'chamfer' in str(model_name):
        results = {"chamfer": results}
    elif 'sinkhorn' in str(model_name):
        results = {"sinkhorn": results}
    elif 'modhdrf' in str(model_name):
        results = {"modhdrf": results}
    elif 'intensity' in str(model_name):
        results = {"intensity": results}
    elif 'dis_pts' in str(model_name):
        results = {"dis_pts": results}
    return results


def modhdrf_dist(pc0_file_path, pc1_file_path, use_smaller_windows=False):
    """
    M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object matching.
    In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    :param pc0_file_path:
    :param pc1_file_path:
    :param use_smaller_windows:
    :return:
    """

    pc_0 = np.asarray(o3d.io.read_point_cloud(pc0_file_path, 'xyz').points)
    pc_1 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)
    A = pc_0[:, :3]
    B = pc_1[:, :3]
    D = cdist(A, B)
    col_mean = np.mean(np.min(D, axis=0))
    row_mean = np.mean(np.min(D, axis=1))

    return max(col_mean, row_mean)


def hausdorff_dist(pc0_file_path, pc1_file_path, use_smaller_windows=False):
    pc_0 = np.asarray(o3d.io.read_point_cloud(pc0_file_path, 'xyz').points)
    pc_1 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)

    x_min = pc_0[:, 0].min()
    x_max = pc_0[:, 0].max()
    y_min = pc_0[:, 1].min()
    y_max = pc_0[:, 1].max()

    if use_smaller_windows:
        step_size = (x_max - x_min) / 3
        x_vec = np.linspace(x_min, x_max + 0.5 * step_size, int((x_max - x_min) / step_size))
        y_vec = np.linspace(y_min, y_max + 0.5 * step_size, int((y_max - y_min) / step_size))
        distVec = []
        for x_ind, x in enumerate(x_vec[:-1]):
            for y_ind, y in enumerate(y_vec[:-1]):
                win0 = pc_0[(pc_0[:, 0] >= x) & (pc_0[:, 0] < x_vec[x_ind + 1]) &
                            (pc_0[:, 1] >= y) & (pc_0[:, 1] < y_vec[y_ind + 1])]
                win1 = pc_1[(pc_1[:, 0] >= x) & (pc_1[:, 0] < x_vec[x_ind + 1]) &
                            (pc_1[:, 1] >= y) & (pc_1[:, 1] < y_vec[y_ind + 1])]

                if win0.shape[0] <= 10 or win1.shape[0] <= 10:  # skip in case a window has small amount of points
                    continue
                window_dist = np.max([directed_hausdorff(win0, win1)[0], directed_hausdorff(win1, win0)[0]])
                distVec.append(window_dist)
        if (distVec == []):
            distVec = 0
        window_dist = np.mean(np.asarray(distVec))  # mean - smaller windows statistics

    else:
        win0 = pc_0[(pc_0[:, 0] >= x_min) & (pc_0[:, 0] < x_max) & (pc_0[:, 1] >= y_min) &
                    (pc_0[:, 1] < y_max)]
        win1 = pc_1[(pc_1[:, 0] >= x_min) & (pc_1[:, 0] < x_max) & (pc_1[:, 1] >= y_min) &
                    (pc_1[:, 1] < y_max)]

        window_dist = np.max([directed_hausdorff(win0, win1)[0], directed_hausdorff(win1, win0)[0]])
    return window_dist


def chamfer_dist(pc0_file_path, pc1_file_path, use_smaller_windows=False):
    pc_0 = np.asarray(o3d.io.read_point_cloud(pc0_file_path, 'xyz').points)
    pc_1 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)

    x_min = pc_0[:, 0].min()
    x_max = pc_0[:, 0].max()
    y_min = pc_0[:, 1].min()
    y_max = pc_0[:, 1].max()

    if use_smaller_windows:
        step_size = (x_max - x_min) / 3
        x_vec = np.linspace(x_min, x_max + 0.5 * step_size, int((x_max - x_min) / step_size))
        y_vec = np.linspace(y_min, y_max + 0.5 * step_size, int((y_max - y_min) / step_size))
        distVec = []
        for x_ind, x in enumerate(x_vec[:-1]):
            for y_ind, y in enumerate(y_vec[:-1]):
                win0 = pc_0[(pc_0[:, 0] >= x) & (pc_0[:, 0] < x_vec[x_ind + 1]) &
                            (pc_0[:, 1] >= y) & (pc_0[:, 1] < y_vec[y_ind + 1])]
                win1 = pc_1[(pc_1[:, 0] >= x) & (pc_1[:, 0] < x_vec[x_ind + 1]) &
                            (pc_1[:, 1] >= y) & (pc_1[:, 1] < y_vec[y_ind + 1])]

                if win0.shape[0] <= 10 or win1.shape[0] <= 10:  # skip in case a window has small amount of points
                    continue
                window_dist = pcu.chamfer(win0, win1)[0]
                distVec.append(window_dist)
        if (distVec == []):
            distVec = 0
        window_dist = np.mean(np.asarray(distVec))  # mean - smaller windows statistics

    else:
        win0 = pc_0[(pc_0[:, 0] >= x_min) & (pc_0[:, 0] < x_max) & (pc_0[:, 1] >= y_min) &
                    (pc_0[:, 1] < y_max)]
        win1 = pc_1[(pc_1[:, 0] >= x_min) & (pc_1[:, 0] < x_max) & (pc_1[:, 1] >= y_min) &
                    (pc_1[:, 1] < y_max)]

        if win0.shape[0] > 50000 or win1.shape[0] > 50000:
            window_dist = 0
        else:
            window_dist = pcu.chamfer(win0, win1)[0]

    return window_dist


def get_z_mean_std(pc0_file_path, pc1_file_path, use_smaller_windows=False):
    pc_0 = np.asarray(o3d.io.read_point_cloud(pc0_file_path, 'xyz').points)
    pc_1 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)
    x_min = pc_0[:, 0].min()
    x_max = pc_0[:, 0].max()
    y_min = pc_0[:, 1].min()
    y_max = pc_0[:, 1].max()

    pc0_std_z = []
    pc1_std_z = []
    pc0_mean_z = []
    pc1_mean_z = []

    if use_smaller_windows:
        step_size = (x_max - x_min) / 3
        x_vec = np.linspace(x_min, x_max + 0.5 * step_size, int((x_max - x_min) / step_size))
        y_vec = np.linspace(y_min, y_max + 0.5 * step_size, int((y_max - y_min) / step_size))
        distVec = []
        for x_ind, x in enumerate(x_vec[:-1]):
            for y_ind, y in enumerate(y_vec[:-1]):
                window_pc0 = pc_0[(pc_0[:, 0] >= x) & (pc_0[:, 0] < x_vec[x_ind + 1]) &
                                 (pc_0[:, 1] >= y) & (pc_0[:, 1] < y_vec[y_ind + 1])]
                window_pc1 = pc_1[(pc_1[:, 0] >= x) & (pc_1[:, 0] < x_vec[x_ind + 1]) &
                                 (pc_1[:, 1] >= y) & (pc_1[:, 1] < y_vec[y_ind + 1])]

                if (window_pc0.shape[0] > 10) and (window_pc1.shape[0] > 10):
                    pc0_std_z.append(np.std(window_pc0[:, 2]))
                    pc1_std_z.append(np.std(window_pc1[:, 2]))
                    pc0_mean_z.append(np.mean(window_pc0[:, 2]))
                    pc1_mean_z.append(np.mean(window_pc1[:, 2]))
        mean_res = np.abs(np.mean(np.asarray(pc0_mean_z)) - np.mean(np.asarray(pc1_mean_z)))
        if np.isnan(float(mean_res)):
            mean_res = 0
        std_res = np.abs(np.mean(np.asarray(pc0_std_z)) - np.mean(np.asarray(pc1_std_z)))
        if np.isnan(float(std_res)):
            std_res = 0
    else:
        window_pc0 = pc_0[(pc_0[:, 0] >= x_min) & (pc_0[:, 0] < x_max) &
                         (pc_0[:, 1] >= y_min) & (pc_0[:, 1] < y_max)]
        window_pc1 = pc_1[(pc_1[:, 0] >= x_min) & (pc_1[:, 0] < x_max) &
                         (pc_1[:, 1] >= y_min) & (pc_1[:, 1] < y_max)]
        mean_res = np.abs(np.mean(window_pc0[:, 2]) - np.mean(window_pc1[:, 2]))
        if np.isnan(float(mean_res)):
            mean_res = 0
        std_res = np.abs(np.std(window_pc0[:, 2]) - np.std(window_pc1[:, 2]))
        if np.isnan(float(std_res)):
            std_res = 0
    return [mean_res, std_res]



def knn_distance(pc0_file_path, pc1_file_path,use_smaller_windows=False):
    pc_0 = np.asarray(o3d.io.read_point_cloud(pc0_file_path, 'xyz').points)
    pc_1 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)
    idx = []
    # gnd = np.percentile(pc_1[:, 2], 2)
    neigh = NearestNeighbors(radius=0.2)
    neigh.fit(pc_0[:, :3])
    for i_x, i in enumerate(pc_1):
        if len(neigh.radius_neighbors(i[:3].reshape(1, -1), radius=0.2, return_distance=False)[0]) == 0:
            # if i[2] < gnd + 3:
            idx.append(i_x)
    print(len(idx))
    return len(idx)

def anomaly_by_intensity(pc0_file_path, pc1_file_path, use_smaller_windows=True):
    optimal_std_thd = 0.05
    overlap = 0.5
    pc_1 = pd.read_csv(pc1_file_path, sep='\s+', header=None).to_numpy()
    pc_1 = pc_1[:,:4]
    # pc_0 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)
    # pc_1 = np.asarray(o3d.io.read_point_cloud(pc1_file_path, 'xyz').points)
    # filter high (outliers) intensity values
    filtered_i_pc_1 = pc_1[(pc_1[:, 3] < intensity_thd)]
    # filtered_i_pc_1 = pc_1[(pc_1[:, 3] < intensity_thd)]
    ground = np.percentile(filtered_i_pc_1[:, 2], 5)
    i_weighted_mean = 0
    x_min = filtered_i_pc_1[:, 0].min()
    x_max = filtered_i_pc_1[:, 0].max()
    y_min = filtered_i_pc_1[:, 1].min()
    y_max = filtered_i_pc_1[:, 1].max()
    # high ground stats for reference - calculated for each window
    leaves_i_mean = np.nanmean(filtered_i_pc_1[filtered_i_pc_1[:, 2] > ground + leaves_height][:, 3])
    leaves_i_std = np.nanstd(filtered_i_pc_1[filtered_i_pc_1[:, 2] > ground + leaves_height][:, 3])
    if filtered_i_pc_1.shape[0] < min_num_points:
        return 0  # not enough points to measure
    # filtering the working window
    filtered_i_pc_1 = filtered_i_pc_1[(filtered_i_pc_1[:, 3] < intensity_thd) & (filtered_i_pc_1[:, 2] > ground) & (
            filtered_i_pc_1[:, 2] < ground + leaves_height)]
    if use_smaller_windows:
        sub_win_size = (x_max - x_min) / 3
        means_vec = []
        stds_vec = []
        x_vec = np.linspace(x_min, x_max + 0.5 * sub_win_size, int((x_max - x_min) / sub_win_size))
        y_vec = np.linspace(y_min, y_max + 0.5 * sub_win_size, int((y_max - y_min) / sub_win_size))
        distVec = []
        for x_ind, x in enumerate(x_vec[:-1]):
            for y_ind, y in enumerate(y_vec[:-1]):
                win0 = pc_1[(pc_1[:, 0] >= x) & (pc_1[:, 0] < x_vec[x_ind + 1]) &
                            (pc_1[:, 1] >= y) & (pc_1[:, 1] < y_vec[y_ind + 1])]
                if win0.shape[0] <= 10:  # skip in case a window has small amount of points
                    continue
                sub_win_mean = np.nanmean(win0[:, 3])
                # sub_win_std = np.nanstd(win0[:, 3])
                means_vec.append(sub_win_mean)
                # stds_vec.append(sub_win_std)
        if (means_vec == []):
            means_vec = 0        # suspicious window based on multiple sub windows decision
        if np.count_nonzero(means_vec > leaves_i_mean + optimal_std_thd * leaves_i_std) >= 2:
            means_vec = np.array(means_vec)
            i_weighted_mean = np.nanmean(means_vec[means_vec > leaves_i_mean + optimal_std_thd * leaves_i_std])
    else:
        win1 = filtered_i_pc_1[(filtered_i_pc_1[:, 0] >= x_min) & (filtered_i_pc_1[:, 0] < x_max) &
                               (filtered_i_pc_1[:, 1] >= y_min) & (filtered_i_pc_1[:, 1] < y_max)]
        # win1 = pc_1[(pc_1[:, 0] > x_min) & (pc_1[:, 0] < x_max) & (pc_1[:, 1] > y_min) &
        #             (pc_1[:, 1] < y_max3)]
        i_weighted_mean = np.nanmean(win1[:, 3])
    print(i_weighted_mean)
    return i_weighted_mean




