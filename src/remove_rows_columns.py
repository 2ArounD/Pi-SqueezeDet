from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import joblib
import numpy as np
import tensorflow as tf

from config import *
from nets import *
from dataset import kitti


class remove_rows_columns(object):
    def __init__(self, threshold,
                 PRETRAINED_MODEL_PATH = '/path/to/parameters/SqueezeDetPlus/SqueezeDetPlus.pkl', #Used to derive structure of network
                 checkpoint_dir = '/path/to/ckpt/checkpoints/pruning/model.ckpt-2400'): #CKPT can replace paramters with the retrained parameters


        #  Create network
        self.mc = kitti_squeezeDetPlus_config()
        self.mc.LOAD_PRETRAINED_MODEL = True
        self.mc.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH
        self.mc.BATCH_SIZE = 1
        self.mc.IS_PRUNING = True
        self.model = SqueezeDetPlusPruneFilterShape(self.mc)

        self.checkpoint_dir = checkpoint_dir
        self.image_set = 'val'
        self.data_path = './data/KITTI'

        self.threshold = threshold
        self.dic_rows_pruned = {}
        self.dic_cols_pruned = {}
        self.masks_values = None

        self.gammas = [par for par in self.model.model_params if 'gamma' in par.name]
        self.kernels = [par for par in self.model.model_params if par.name[-9:-2]=='kernels']
        self.biases = [par for par in self.model.model_params if par.name[-8:-2]=='biases']

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            self.restore_checkpoint(sess)
            self.masks_values = sess.run(self.gammas)
            self.remove_rows_columns(sess, threshold)

        tf.reset_default_graph()

    def remove_rows_columns(self, sess, threshold):
        #get old parameters dictionary for slicer variable
        assert tf.gfile.Exists(self.mc.PRETRAINED_MODEL_PATH), \
        'Cannot find pretrained model at the given path:' \
        '  {}'.format(self.mc.PRETRAINED_MODEL_PATH)
        weights_dic_old = joblib.load(self.mc.PRETRAINED_MODEL_PATH)

        weights = tf.global_variables()
        # Dictionary to store parameters to create pruned network
        dic = {}
        entry = []
        remove_col = None
        remove_row = None
        #Slicer variable to store offset of pruned filter
        slicer = [[0, 0], [0, 0]]

        for var_i, var in enumerate(weights):
            # logic to check if row of next kernel needs removing
            if 'gamma_row:0' in var.name:
                mask_row = sess.run(var)
                if min(abs(mask_row[0,0]), abs(mask_row[-1,-1])) < threshold:
                    if abs(mask_row[0,0]) < abs(mask_row[-1,-1]):
                        remove_row = 0
                    else:
                        remove_row = -1
                dic[var.name[:-11] + '_mask_row'] = mask_row

            # logic to check if column of next kernel needs removing
            if 'gamma_col:0' in var.name:
                mask_col = sess.run(var)
                if min(abs(mask_col[0,0]), abs(mask_col[-1,-1])) < threshold:
                    if abs(mask_col[0,0]) < abs(mask_col[-1,-1]):
                        remove_col = 0
                    else:
                        remove_col = -1
                dic[var.name[:-11] + '_mask_col'] = mask_col

            # Remove rows and column according to remove_row and remove_col booleans
            if 'kernels:0' in var.name:
                kernel = sess.run(var)
                kernel_pruned = kernel
                if 'conv1/kernels:0' in var.name or 'expand3x3/kernels:0' in var.name:
                    mask_col = sess.run(weights[var_i+3])
                    mask_row = sess.run(weights[var_i+2])
                    # Load slicer variable if layer has had rows or cols removed before
                    if var.name[:-10] + '_slicer' in weights_dic_old.keys():
                        slicer = weights_dic_old[var.name[:-10] + '_slicer']
                    else:
                        slicer = [[0,0],[0,0]]

                    #Multiply kernel with row factros
                    kernel_pruned = np.transpose(np.matmul(np.expand_dims(
                                                 np.expand_dims(mask_row, 0), 0),
                                                 np.transpose(kernel)))

                    #Multiply kernel with col factros
                    kernel_pruned = np.transpose(np.matmul(np.transpose(kernel_pruned),
                                        np.expand_dims(np.expand_dims(mask_col, 0), 0)))

                    #If row needs to be removed, remove and store offset in slicer
                    if remove_row != None:
                        self.dic_rows_pruned[var.name[:-10]] = 1
                        mask_row[remove_row][remove_row] = 0
                        kernel_pruned = np.delete(kernel_pruned, remove_row, 1)
                        if remove_row == 0:
                            slicer[0][1] = slicer[0][1] + 1
                        else:
                            slicer[1][1] = slicer[1][1] + 1
                        remove_row = None
                    else:
                        self.dic_rows_pruned[var.name[:-10]] = 0

                    #If col needs to be removed, remove and store offset in slicer
                    if remove_col != None:
                        mask_col[remove_col][remove_col] = 0
                        self.dic_cols_pruned[var.name[:-10]] = 1
                        kernel_pruned = np.delete(kernel_pruned, remove_col, 0)
                        if remove_col == 0:
                            slicer[0][0] = slicer[0][0] + 1
                        else:
                            slicer[1][0] = slicer[1][0] + 1
                        remove_col = None
                    else:
                        self.dic_cols_pruned[var.name[:-10]] = 0

                    dic[var.name[:-10]+'_slicer'] = slicer
                    slicer = [[0,0],[0,0]]

                if kernel_pruned.shape[0] > 1:
                    kernel_new = np.zeros((kernel_pruned.shape[3],
                                          kernel_pruned.shape[2],
                                          kernel_pruned.shape[0],
                                          kernel_pruned.shape[1],
                                          ))
                    for x in range(kernel_pruned.shape[0]):
                        for y in range(kernel_pruned.shape[1]):
                            kernel_new.T[y][x] = kernel_pruned[x][y]
                    kernel_pruned = kernel_new
                else:
                    kernel_pruned = kernel_pruned.T

            if 'biases:0' in var.name:
                biases_pruned = sess.run(var)
                entry = [kernel_pruned, biases_pruned]
                dic[var.name[:-9]] = entry
                entry = []

        joblib.dump(dic,'SqueezeDetPruned1.pkl',compress=False )

    def restore_checkpoint(self, sess):
        saver = tf.train.Saver(self.model.model_params)
        if os.path.isfile(self.checkpoint_dir + '.meta'):
          saver.restore(sess, self.checkpoint_dir)

Pruner = remove_rows_columns(0.1)
