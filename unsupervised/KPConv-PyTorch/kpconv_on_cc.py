#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

class CC_based_modelnet40(PointCloudDataset):
    """Class to handle Modelnet 40 dataset."""

    def __init__(self, config, train=True, orient_correction=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'ModelNet40')

        ############
        # Parameters
        ############

        # Dict from labels to names

        self.label_to_names = {0: 'wardrobe',
                       1: 'table',
                       2: 'door',
                       3: 'car',
                       4: 'tv_stand',
                       5: 'sofa',
                       6: 'tent',
                       7: 'glass_box',
                       8: 'desk',
                       9: 'sitting_person',
                       10: 'night_stand',
                       11: 'bathtub',
                       12: 'person',
                       13: 'mantel',
                       14: 'cup',
                       15: 'plant',
                       16: 'chair',
                       17: 'bed',
                       18: 'dresser',
                       19: 'bench'}


        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = ''

        # Type of task conducted on this dataset
        self.dataset_task = 'classification'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.train = train

        # Number of models and models used per epoch
        if self.train:
            self.num_models = 9843
            if config.epoch_steps and config.epoch_steps * config.batch_num < self.num_models:
                self.epoch_n = config.epoch_steps * config.batch_num
            else:
                self.epoch_n = self.num_models
        else:
            # self.num_models = 2468
            self.num_models = len(glob.glob(self.path+'/*_*.txt',recursive=True)) -1 # to bechanged num of vals
            self.epoch_n = min(self.num_models, config.validation_size * config.batch_num)

        #############
        # Load models
        #############

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        self.input_points, self.input_normals, self.input_labels = self.load_subsampled_clouds(orient_correction)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_models

    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        ###################
        # Gather batch data
        ###################

        tp_list = []
        tn_list = []
        tl_list = []
        ti_list = []
        s_list = []
        R_list = []

        for p_i in idx_list:
            # Get points and labels
            points = self.input_points[p_i].astype(np.float32)
            normals = self.input_normals[p_i].astype(np.float32)
            # input_labels = np.array([24 for name in self.input_labels])
            # label = self.label_to_idx[input_labels[p_i]]
            label = self.label_to_idx[self.input_labels[p_i]]

            # Data augmentation
            points, normals, scale, R = self.augmentation_transform(points, normals)

            # Stack batch
            tp_list += [points]
            tn_list += [normals]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [scale]
            R_list += [R]

        ###################
        # Concatenate batch
        ###################

        # show_ModelNet_examples(tp_list, cloud_normals=tn_list)

        stacked_points = np.concatenate(tp_list, axis=0)
        stacked_normals = np.concatenate(tn_list, axis=0)
        labels = np.array(tl_list, dtype=np.int64)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, stacked_normals))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.classification_inputs(stacked_points,
                                                stacked_features,
                                                labels,
                                                stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, model_inds]

        return input_list

    def load_subsampled_clouds(self, orient_correction):

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        if self.train:
            split = 'training'
        else:
            split = 'test'

        print('\nLoading {:s} points subsampled at {:.3f}'.format(split, self.config.first_subsampling_dl))
        filename = join(self.path, '{:s}_{:.3f}_record.pkl'.format(split, self.config.first_subsampling_dl))

        if exists(filename):
            with open(filename, 'rb') as file:
                input_points, input_normals, input_labels = pickle.load(file)

        # Else compute them from original points
        else:

            # # Collect training file names
            # if self.train:
            #     names = np.loadtxt(join(self.path, 'modelnet40_train.txt'), dtype=np.str)

            names = np.loadtxt(join(self.path, 'cc_test.txt'), dtype=np.str)

            # Initialize containers
            input_points = []
            input_normals = []

            # Advanced display
            N = len(names)
            progress_n = 30
            fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'

            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                # txt_file = join(self.path, class_folder, cloud_name) + '.txt'
                txt_file = join(self.path, cloud_name)
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if self.config.first_subsampling_dl > 0:
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampleDl=self.config.first_subsampling_dl)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                print('', end='\r')
                print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

                # Add to list
                input_points += [points]
                input_normals += [normals]

            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
            print()

            # Get labels
            # label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            label_names = ['bed' for name in names]
            # label_names[np.where('person1.txt' == names)[0].ravel()[0]] = 'person'
            # label_names[np.where('person2.txt' == names)[0].ravel()[0]] = 'person'
            input_labels = np.array([self.name_to_label[name] for name in label_names])
            # for idx,name in enumerate(names):
            #
            #     print(f'{idx} {name}\n ')
            # input_labels = np.array([name for name in label_names])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((input_points,
                             input_normals,
                             input_labels), file)

        lengths = [p.shape[0] for p in input_points]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        if orient_correction:
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]
            input_normals = [nn[:, [0, 2, 1]] for nn in input_normals]

        return input_points, input_normals, input_labels

class CC_based_modelnet40_sampler(Sampler):
    """Sampler for ModelNet40"""

    def __init__(self, dataset: ModelNet40Dataset, use_potential=True, balance_labels=False):
        Sampler.__init__(self, dataset)

        # Does the sampler use potential for regular sampling
        self.use_potential = use_potential

        # Should be balance the classes when sampling
        self.balance_labels = balance_labels

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = None

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = 10000

        return

    def __iter__(self):
        """
        Yield next batch indices here
        """

        ##########################################
        # Initialize the list of generated indices
        ##########################################

        if self.use_potential:
            if self.balance_labels:

                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                for i, l in enumerate(self.dataset.label_values):

                    # Get the potentials of the objects of this class
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    class_potentials = self.potentials[label_inds]

                    # Get the indices to generate thanks to potentials
                    if pick_n < class_potentials.shape[0]:
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = label_inds[pick_indices]
                    gen_indices.append(class_indices)

                # Stack the chosen indices of all classes
                gen_indices = np.random.permutation(np.hstack(gen_indices))

            else:

                # Get indices with the minimum potential
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[:self.dataset.epoch_n]
                else:
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)

            # Update potentials (Change the order for the next epoch)
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                gen_indices = []
                for l in self.dataset.label_values:
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                    gen_indices += [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_i in gen_indices:

            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_i]

            # Update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)

        return 0

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return None

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = False

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                   self.dataset.config.batch_num)
        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.conv_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in
                              batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                          neighb_hists[layer, neighb_size],
                                                          bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                       self.dataset.config.batch_num)
            batch_lim_dict[key] = self.batch_limit
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return
def model_choice(chosen_log):
    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'results/Log_2020-04-05_19-19-20'  # => ModelNet40
    chosen_log = '/data/Carmel/KPConv-PyTorch-master/results/Log_2021-02-25_11-23-33'
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, '')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        # chosen_chkp = 'current_chkp.tar'
        chosen_chkp = 'chkp_0350.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, '', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 0.0001
    # config.augment_symmetries = False
    # config.batch_num = 3
    # config.in_radius = 4
    config.validation_size = 200
    # config.input_threads = 10
    config.input_threads = 1


    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        test_dataset = CC_based_modelnet40(config, train=False)
        test_sampler = CC_based_modelnet40_sampler(test_dataset)
        collate_fn = ModelNet40Collate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        probs = tester.classification_test(net, test_loader, config)
        save_probs_dir = ''
        np.savetxt(save_probs_dir + 'probs.txt', probs)
    # elif config.dataset_task == 'cloud_segmentation':
    #     tester.cloud_segmentation_test(net, test_loader, config)
    # elif config.dataset_task == 'slam_segmentation':
    #     tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
