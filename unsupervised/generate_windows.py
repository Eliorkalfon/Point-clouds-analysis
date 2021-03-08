from pathlib import Path
import json
import itertools
import pandas as pd
import concurrent.futures
import pickle
import numpy as np
import os
import glob
import laspy
from sklearn.neighbors import NearestNeighbors

MAX_POINTS = 80000
N_POINTS = 500

XSTEP = 3
YSTEP = 3
MIN_INITIAL_POINTS = int(XSTEP * 100)  # assuming at least 50 points per window
MIN_REDUCED_POINTS = 100  # 100


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


def remove_close_to_sensor(pc, x=0, y=0, z=0, dis=50):
    idx = []
    ref_pts = np.array([x, y, z])
    for i in range(pc.shape[0]):
        if np.linalg.norm(pc[i, :3] - ref_pts) > dis:
            idx.append(i)
    pc_new = np.copy(pc[idx])
    return pc_new


def remove_far_to_sensor(pc, x=0, y=0, z=0, dis=300):
    idx = []
    ref_pts = np.array([x, y, z])
    for i in range(pc.shape[0]):
        if np.linalg.norm(pc[i, :3] - ref_pts) < dis:
            idx.append(i)
    pc_new = np.copy(pc[idx])
    return pc_new


def get_sliding_windows(x_min, x_max, y_min, y_max, overlap=0.0):
    """
    receives the bounding box of a scene, then separates it into evenly spaced x and y vectors - by the X/Y steps
    :return array containing every pair of xy (describing the min x,y of every window inside the scene)
    :param overlap: percent of overlap wanted between the windows
    """
    if overlap == 0:
        overlap = 1
    overlap = 1 / overlap
    # x_min += 10
    # y_min += 10
    # x_max -= 10
    # y_max -= 10
    x_vec = np.linspace(x_min, x_max + 0.5 * XSTEP, int(((x_max - x_min) / XSTEP) * overlap))
    y_vec = np.linspace(y_min, y_max + 0.5 * YSTEP, int(((y_max - y_min) / YSTEP) * overlap))

    all_coords = list(itertools.product(x_vec, y_vec))

    return all_coords, x_vec, y_vec


def voxelization_pc(pc, voxel_size=0.05):
    points = pc[:, :3]
    nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)
    voxel_grid = {}
    grid_barycenter, grid_candidate_center = [], []
    last_seen = 0
    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = pc[idx_pts_vox_sorted[
                                    last_seen:last_seen + nb_pts_per_voxel[idx]]]
        grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)][0][:3] -
                                                                           np.mean(voxel_grid[tuple(vox)][0][:3],
                                                                                   axis=0),
                                                                           axis=0).argmin()])
        last_seen += nb_pts_per_voxel[idx]
    return np.array(grid_candidate_center)


class SaveWindows:
    def __init__(self, params):
        self.layers = params['layers']
        self.out_path = f"{params['data_dir']}/{params['out_dir']}"
	self.pc_1_dir = glob.glob(str(params['data_dir']) + f'/*t{params["pc_1"]}.*', recursive=True)[0]

        os.makedirs(f'{self.out_path}/pc{self.pc_1_dir[-5]}', exist_ok=True)  # create pc_1 folder
        self.tagging_method = params['tagging_method']
        self.load_dataset(self.pc_1_dir)  # load dataset into self.dataset
        x_min = self.dataset[:, 0].min()
        x_max = self.dataset[:, 0].max()
        y_min = self.dataset[:, 1].min()
        y_max = self.dataset[:, 1].max()
        self.all_coords, self.x_vec, self.y_vec = get_sliding_windows(x_min, x_max, y_min, y_max, 0)
        self.coords_pairs = []
        json.dump({'coords': self.all_coords}, open(f'{self.out_path}/pc_0_list_of_bboxes.json', 'w'))

    def load_dataset(self, data_path):
        self.scene_idx = data_path[-5]
        suffix = Path(data_path).suffix
        if suffix == '.las':
            las_data = laspy.file.File(file_path)
            additional_filters = ''
            for layer in self.layers:
                additional_filters += 'las_data.' + layer + ','
            self.dataset = eval(f'np.column_stack([las_data.x, las_data.y, las_data.z, {additional_filters}])')

        elif suffix == '.txt':
            # txt_data = np.loadtxt(data_path)
            # txt_data = np.loadtxt(data_path)
            txt_data = pd.read_csv(data_path, sep='\s+', header=None).to_numpy()
            # Normalize intensity
            txt_data[:, 3] = (txt_data[:, 3] - np.amin(txt_data[:, 3])) / (
                    np.amax(txt_data[:, 3]) - np.amin(txt_data[:, 3]))
            # txt_data = remove_far_to_sensor(txt_data)
            # txt_data = remove_close_to_sensor(txt_data, dis=50)
            additional_filters = ''
            for i in range(len(self.layers)):
                additional_filters += 'txt_data[:,' + str(i + 3) + '],'
            self.dataset = eval(
                f'np.column_stack([txt_data[:, 0], txt_data[:, 1], txt_data[:, 2], {additional_filters}])')

        elif suffix == '.pcd':
            pcd_data = np.loadtxt(data_path, skiprows=10)
            additional_filters = ''
            for i in range(len(self.layers)):
                additional_filters += 'pcd_data[:,' + str(i + 3) + '],'
            self.dataset = eval(
                f'np.column_stack([pcd_data[:, 0], pcd_data[:, 1], pcd_data[:, 2], {additional_filters}])')

    def save_window(self, coord):
        filename = 0
        suspicious = 0
        x, y, next_x, next_y = coord
        window = self.dataset[(self.dataset[:, 0] >= x) & (self.dataset[:, 0] < next_x) &
                              (self.dataset[:, 1] >= y) & (self.dataset[:, 1] < next_y)]

        # window =window[((window[:, 0] >= X_MIN) & (window[:, 0] <= X_MAX) & (window[:, 1] >= Y_MIN) & (
        #         window[:, 1] <= Y_MAX))]
        v_s = 0.05
        win_size = window.shape[0]
        if win_size > 20000:
            if win_size > 50000:
                v_s = 0.2
            elif win_size > 40000:
                v_s = 0.15
            elif win_size > 30000:
                v_s = 0.1
            window = voxelization_pc(window, voxel_size=v_s)

        if window.shape[0] >= MIN_INITIAL_POINTS:  # make sure a window has points in it
            window = remove_pc_outliers(window, radius=0.5, pts=1)
            if window.shape[0] >= MIN_REDUCED_POINTS:
                suspicious = ''
                n_points_in_window = 0

                window_label = 0
                x_min = window[:, 0].min()
                x_max = window[:, 0].max()
                y_min = window[:, 1].min()
                y_max = window[:, 1].max()
                z_min = window[:, 2].min()
                z_max = window[:, 2].max()
                filename = f'{self.out_path}/t{self.scene_idx}/{x}_{y}_{suspicious}_{n_points_in_window}_' \
                           f'{window.shape[0]}_{window_label}.txt'
                np.savetxt(filename, window)
        return filename, suspicious

    def save_preprocess_metadata(self, pc_0_filenames, pc_1_filenames, y_label):
        meta = {'layers': [i for i in self.layers],
                'step_size': XSTEP,
                'files_location': self.out_path,
                '# pc_0 files': len(os.listdir(f'{self.out_path}/t{self.pc_0_dir[-5]}')),
                '# pc_1 files': len(os.listdir(f'{self.out_path}/t{self.pc_1_dir[-5]}/')),
                'tagging_method': self.tagging_method}

        # prepare spatial matrix
        df = pd.DataFrame(columns=['xy'])
        df['xy'] = self.coords_pairs
        meta['spatial_matrix'] = df.to_numpy().reshape(self.x_vec.shape[0] - 1, self.y_vec.shape[0] - 1, 1)

        # prepare file names matrix preparation
        pc_0_dropped_win = pc_0_filenames[pc_0_filenames == 0]
        pc_1_dropped_win = pc_1_filenames[pc_1_filenames == 0]

        pc_0_filenames[pc_1_dropped_win.index] = 0
        pc_1_filenames[pc_0_dropped_win.index] = 0

        pc_0_indices = pc_0_filenames[pc_0_filenames != 0].index
        y_label = y_label[pc_0_indices]

        meta['pc_0_filenames'] = pc_0_filenames[pc_0_filenames != 0].to_list()
        meta['pc_1_filenames'] = pc_1_filenames[pc_1_filenames != 0].to_list()

        # reshape filenames to spatial matrix shape
        pc_0_filenames = pc_0_filenames.to_numpy().reshape(self.x_vec.shape[0] - 1, self.y_vec.shape[0] - 1, 1)
        pc_1_filenames = pc_1_filenames.to_numpy().reshape(self.x_vec.shape[0] - 1, self.y_vec.shape[0] - 1, 1)

        pc_0_indices = np.where(pc_0_filenames != 0)
        pc_1_indices = np.where(pc_1_filenames != 0)

        meta['pc_0_indices'] = pc_0_indices
        meta['pc_1_indices'] = pc_1_indices

        pickle.dump(meta, open(self.out_path + '/metadata.json', 'wb'))


def generate_windows(params, num_workers=32):
    """
    saves all windows contained in a scene' by the given parameters
    :param num_workers: number of threads
    :param out_folder: folder name for the new data
    :param data_type: see / elbit ...
    :return:
    """
    sw = SaveWindows(params=params)
    coords_pairs = []
    for x_ind, x in enumerate(sw.x_vec[:-1]):
        for y_ind, y in enumerate(sw.y_vec[:-1]):
            coords_pairs.append((x, y, sw.x_vec[x_ind + 1], sw.y_vec[y_ind + 1]))
    sw.coords_pairs = coords_pairs
 
    #  run pc_1
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        print('starting threads')
        pc_1_filenames = []
        pc_1_y_label = []
        for i in executor.map(sw.save_window, coords_pairs):
            pc_1_filenames.append(i[0])
            pc_1_y_label.append(i[1])


if __name__ == '__main__':
    file_path = ''
    params = {'layers': ['intensity', 'reflection'],
              'out_dir': f'windows_{XSTEP}_{YSTEP}', 'data_dir': file_path, 'tagging_method': 'segment', 'pc_0': '0',
              'pc_1': '1'}
    generate_windows(params, num_workers=32)
