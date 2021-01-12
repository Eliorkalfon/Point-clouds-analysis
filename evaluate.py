from * import roc_auc_results  as results
from models.models import *
import numpy as np
import pickle


def predict(path, models, y_labels, clf_majority_prc=1):
    config = pickle.load(open(path + '/metadata.json', 'rb'))
    pc0_filenames, pc1_filenames, spatial_matrix2d, filenames_indices = config['pc0_filenames'], \
                                                                        config['pc1_filenames'], \
                                                                        config['spatial_matrix'], \
                                                                        config['pc0_indices']

    features = {}
    for model in models:
        if 'intensity' in model.__name__ or 'std' in model.__name__ or 'mean' in model.__name__:
            use_smaller_windows = True
        else:
            use_smaller_windows = False
        features.update(multi_proc(pc0_filenames, pc1_filenames, model, use_smaller_windows))
        print('done')
    anomalies_3d_matrix_by_clf = np.zeros(shape=(spatial_matrix2d.shape[0], spatial_matrix2d.shape[1],
                                                 len(features.keys())))
    for i_clf, key in enumerate(features.keys()):
        matrix2d = np.zeros(shape=(spatial_matrix2d.shape[0], spatial_matrix2d.shape[1], 1))
        clf_predictions = results.get_results_wo_clf(features[key], y_labels, roc=True, conf_matrix=True,thd=False)
        matrix2d[filenames_indices] = clf_predictions
        anomalies_3d_matrix_by_clf[:, :, i_clf] = np.squeeze(matrix2d)

    labels_2d = np.zeros(shape=(spatial_matrix2d.shape[0], spatial_matrix2d.shape[1], 1))
    labels_2d[filenames_indices] = y_labels
    nonzero_locations = np.where(np.count_nonzero(anomalies_3d_matrix_by_clf, axis=2) >
                                 anomalies_3d_matrix_by_clf.shape[2] * clf_majority_prc)
    anomalies = anomalies_3d_matrix_by_clf.sum(axis=2)
    anomalies = anomalies[:, :, np.newaxis]
    anomalies[np.where(anomalies < clf_majority_prc * anomalies_3d_matrix_by_clf.shape[2])] = 0
    anomalies[np.where(anomalies > 0)] = 1
    results.get_results_wo_clf(anomalies[filenames_indices], y_labels, roc=True, conf_matrix=True)
    # windows_predictions = anomalies_3d_matrix_by_clf[nonzero_locations]
    return anomalies


def show_mat(win_map, vmax=1, extent=None):
    plt.imshow(win_map, vmin=0, vmax=vmax, extent=extent)  # ,extent=extent)

    # plt.title('Anomalies windows matrix', fontsize=4)
    # plt.figure(figsize=(3, 3))

    # plt.xlabel('X-CORD')
    # plt.ylabel('Y-CORD')
    plt.show()
    # plt.savefig(scenes_dir + f'result_mat_{step}sec')
    # fig, (ax) = plt.subplots(figsize=(8, 4), ncols=1)


def x_y_dim(f_names):
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for f_name in f_names:
        x = float(f_name.split('/')[-1].split('_')[0])
        y = float(f_name.split('/')[-1].split('_')[1])
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    return x_min, x_max, y_min, y_max


if __name__ == '__main__':
    path = ''  # windows path
    lables_path = ''  # true lables
    models = [hausdorff_dist, get_z_mean_std, chamfer_dist,
              anomaly_by_intensity, modhdrf_dist, dis_pts]
    ids = [0, 1, 2, 3, 4, 5]
    models = [models[i] for i in ids]
    y_labels = np.load(lables_path)
    multi_clf = predict(path, models, y_labels, clf_majority_prc=1)
    config = pickle.load(open(path + '/metadata.json', 'rb'))
    pc0_filenames, pc1_filenames, spatial_matrix2d, filenames_indices = config['pc0_filenames'], \
                                                                        config['pc1_filenames'], \
                                                                        config['spatial_matrix'], \
                                                                        config['pc0_indices']
    multi_clf = multi_clf[:, :, 0]
    x_min, x_max, y_min, y_max = x_y_dim(pc1_filenames)
    extent = [x_max, x_min, y_max, y_min]
    show_mat(multi_clf.T, vmax=1, extent=extent)
