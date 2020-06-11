from __future__ import division
from __future__ import print_function

import os

import joblib
import numpy as np
import tensorflow as tf

from config import kitti_squeezeDetPlus_config
from nets import SqueezeDetPlusPruneFilter

threshold = 0.1


class remove_filters(object):
    def __init__(self, threshold,
                 PRETRAINED_MODEL_PATH = '/path/to/weights/SqueezeDetPlus.pkl', #Used to derive structure of network
                 checkpoint_dir = 'path/to/pruning/ckpt/model.ckpt-200' #CKPT can replace paramters with the retrained parameters'
                 ):

        # create model
        self.mc = kitti_squeezeDetPlus_config()
        self.mc.LOAD_PRETRAINED_MODEL = True
        self.mc.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH
        self.mc.BATCH_SIZE = 1
        self.mc.IS_PRUNING = True
        self.mc.LITE_MODE = False
        self.model = SqueezeDetPlusPruneFilter(self.mc)

        self.checkpoint_dir = checkpoint_dir
        self.image_set = 'val'
        self.data_path = './data/KITTI'

        self.threshold = threshold
        self.dic_layers_pruned = {}
        self.dic_n_filters = {}
        self.gammas_values = None

        self.gammas = [par for par in self.model.model_params if 'gamma' in par.name]
        self.kernels = [par for par in self.model.model_params if par.name[-9:-2]=='kernels']
        self.biases = [par for par in self.model.model_params if par.name[-8:-2]=='biases']

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            self.restore_checkpoint(sess)
            self.gammas_values = sess.run(self.gammas)
            self.set_zeros(sess)
            self.remove_filters_and_channels(sess)

        tf.reset_default_graph()

    # Removes all filters and channels set to zero
    def remove_filters_and_channels(self,sess):
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)

        sess.run(init_new_vars_op)

        weights = tf.global_variables()

        #dictionary to store new paramters with layer + var name as key
        dic = {}
        entry = []

        for var in weights:
            if 'kernels:0' in var.name:
                kernel = sess.run(var)
                n_c_in_gone = np.sum(np.all(kernel[0][0] == 0, axis =1))
                n_c_out_gone = np.sum(np.all(kernel[0][0] == 0, axis =0))
                if n_c_in_gone > 0 or n_c_out_gone > 0:
                    kernel_pruned = np.empty([kernel.shape[0], kernel.shape[1],kernel.shape[2] - n_c_in_gone ,kernel.shape[3] - n_c_out_gone])
                    for dim1 in range(0, kernel.shape[0]):
                        for dim2 in range(0,kernel.shape[1]):
                            kernel_pruned[dim1][dim2] = kernel[dim1][dim2][~np.all(kernel[dim1][dim2] == 0, axis =1)].T[~np.all(kernel[dim1][dim2] == 0, axis =0)].T
                else:
                    kernel_pruned = kernel

                if kernel_pruned.shape[0] > 1:
                    kernel_new = np.zeros(kernel_pruned.T.shape)
                    for x in range(kernel_pruned.shape[0]):
                        for y in range(kernel_pruned.shape[1]):
                            kernel_new.T[x][y] = kernel_pruned[y][x]
                    kernel_pruned = kernel_new
                else:
                    kernel_pruned = kernel_pruned.T


            if 'biases:0' in var.name:
                biases = sess.run(var)
                biases_pruned = biases[biases != 0]
                entry = [kernel_pruned, biases_pruned]
                dic[var.name[:-9]] = entry
                entry = []

            if 'gamma_filter:0' in var.name:
                gammas = sess.run(var)
                n_g_gone = np.sum(gammas == 0)
                self.dic_layers_pruned[var.name[:-8]] = n_g_gone
                self.dic_n_filters[var.name[:-8]] = gammas.size
                gammas_pruned = gammas[gammas!=0]
                dic[var.name[:-8] + '_gamma'] = gammas_pruned

        joblib.dump(dic,'SqueezeDetPrunedFilters.pkl',compress=False )

    # Function to loop through gammas and set zeros in the network
    def set_zeros(self, sess):

        pruned_channels = 0
        total_channels = 0
        for idx, gamma in enumerate(self.gammas):
            gam_values = sess.run(gamma)
            mask = np.where(gam_values<self.threshold)
            gam_values[mask] = 0
            ass_op_g = tf.assign(self.gammas[idx], gam_values)
            sess.run(ass_op_g)

            # check layer of gamma and select appropriate function
            if 'conv1/' in gamma.name:
                self.set_zeros_conv1(sess, mask, idx)

            elif 'fire' in gamma.name:
                if 'squeeze1x1' in gamma.name:
                    self.set_zeros_squeeze1x1(sess, mask, idx)
                if 'expand1x1' in gamma.name:
                    self.set_zeros_expand1x1(sess, mask, idx)
                if 'expand3x3' in gamma.name:
                    self.set_zeros_expand3x3(sess, mask, idx)

            pruned_channels = pruned_channels + len(mask[0])
            total_channels = total_channels + len(gam_values)

        print('total_channels: ' + str(total_channels))
        print('channels pruned: ' + str(pruned_channels))

    # Function to set zeros in first conv layer and connected channels
    def set_zeros_conv1(self, sess, mask, gamma_idx):
        kern_values_0 = sess.run(self.kernels[gamma_idx])
        kern_values_1 = sess.run(self.kernels[gamma_idx+1])
        bias_values_0 = sess.run(self.biases[gamma_idx])

        kern_values_1[0][0][:][mask] = 0

        for dim1 in range(0, kern_values_0.shape[0]):
            for dim2 in range(0, kern_values_0.shape[1]):
                kern_values_0[dim1][dim2].T[mask] = 0

        bias_values_0[mask] = 0

        ass_ops = [tf.assign(self.kernels[gamma_idx], kern_values_0),
                   tf.assign(self.kernels[gamma_idx+1], kern_values_1),
                   tf.assign(self.biases[gamma_idx], bias_values_0)]

        sess.run(ass_ops)
        a=0

    # Function to set zeros in s1x1 layer and connected channels
    def set_zeros_squeeze1x1(self, sess, mask, gamma_idx):
        kern_values_0 = sess.run(self.kernels[gamma_idx])
        kern_values_1 = sess.run(self.kernels[gamma_idx+1])
        kern_values_2 = sess.run(self.kernels[gamma_idx+2])
        bias_values_0 = sess.run(self.biases[gamma_idx])

        kern_values_0[0][0].T[mask] = 0
        kern_values_1[0][0][:][mask] = 0

        for dim1 in range(0, kern_values_2.shape[0]):
            for dim2 in range(0, kern_values_2.shape[1]):
                kern_values_2[dim1][dim2][:][mask] = 0

        bias_values_0[mask] = 0

        ass_ops = [tf.assign(self.kernels[gamma_idx], kern_values_0),
                   tf.assign(self.kernels[gamma_idx+1], kern_values_1),
                   tf.assign(self.kernels[gamma_idx+2], kern_values_2),
                   tf.assign(self.biases[gamma_idx], bias_values_0)]

        sess.run(ass_ops)

    # Function to set zeros in e1x1 layer and connected channels
    def set_zeros_expand1x1(self, sess, mask, gamma_idx):
        kern_values_0 = sess.run(self.kernels[gamma_idx])
        kern_values_2 = sess.run(self.kernels[gamma_idx+2])
        bias_values_0 = sess.run(self.biases[gamma_idx])

        kern_values_0[0][0].T[mask] = 0

        for dim1 in range(0, kern_values_2.shape[0]):
            for dim2 in range(0, kern_values_2.shape[1]):
                kern_values_2[dim1][dim2][:][mask] = 0

        bias_values_0[mask] = 0

        ass_ops = [tf.assign(self.kernels[gamma_idx], kern_values_0),
                   tf.assign(self.kernels[gamma_idx+2], kern_values_2),
                   tf.assign(self.biases[gamma_idx], bias_values_0)]

        sess.run(ass_ops)

    # Function to set zeros in e3x3 layer and connected channels
    def set_zeros_expand3x3(self, sess, mask, gamma_idx):
        kern_values_min1 = sess.run(self.kernels[gamma_idx-1])
        kern_values_0 = sess.run(self.kernels[gamma_idx])
        kern_values_1 = sess.run(self.kernels[gamma_idx+1])
        bias_values_0 = sess.run(self.biases[gamma_idx])

        for dim1 in range(0, kern_values_0.shape[0]):
            for dim2 in range(0, kern_values_0.shape[1]):
                kern_values_0[dim1][dim2].T[mask] = 0

        for dim1 in range(0, kern_values_1.shape[0]):
            for dim2 in range(0, kern_values_1.shape[1]):
                kern_values_1[dim1][dim2][:][mask[0]+kern_values_min1.shape[3]] = 0


        bias_values_0[mask] = 0

        ass_ops = [tf.assign(self.kernels[gamma_idx], kern_values_0),
                   tf.assign(self.kernels[gamma_idx+1], kern_values_1),
                   tf.assign(self.biases[gamma_idx], bias_values_0)]

        sess.run(ass_ops)

    # Restore checkpoint with pruning variables
    def restore_checkpoint(self, sess):
        saver = tf.train.Saver(self.model.model_params)

        if os.path.isfile(self.checkpoint_dir + '.meta'):
          saver.restore(sess, self.checkpoint_dir)

Pruner = remove_filters(threshold)
