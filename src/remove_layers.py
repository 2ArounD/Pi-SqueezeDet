from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import joblib
import numpy as np
import tensorflow as tf
import pdb

from config import *
from nets import *


class l1_model_pruner(object):
    def __init__(self, threshold,
                 PRETRAINED_MODEL_PATH = '/path/to/parameters/SqueezeDetPlus/SqueezeDetPlus.pkl', #Used to derive structure of network
                 checkpoint_dir = '/path/to/ckpt/checkpoints/pruning/model.ckpt-2400'): #CKPT can replace paramters with the retrained parameters


        # Create network
        self.mc = kitti_squeezeDetPlus_config()
        self.mc.LOAD_PRETRAINED_MODEL = True
        self.mc.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH
        self.mc.BATCH_SIZE = 10
        self.mc.IS_PRUNING = True
        self.model = SqueezeDetPlusPruneLayer(self.mc)

        self.checkpoint_dir = checkpoint_dir
        self.image_set = 'val'
        self.data_path = '../data/KITTI'

        self.threshold = threshold
        self.lay_gammas = np.array([])
        self.dic_layers_pruned = {}

        self.gammas = [par for par in self.model.model_params if par.name[-13:-2]=='layer_gamma']
        self.kernels = [par for par in self.model.model_params if par.name[-9:-2]=='kernels']
        self.biases = [par for par in self.model.model_params if par.name[-8:-2]=='biases']

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            self.restore_checkpoint(sess)
            self.remove_layers(sess, threshold)
        tf.reset_default_graph()

    # Remove layers from network and store information in dictionary
    def remove_layers(self,sess, threshold):

        weights = tf.global_variables()
        #dictionary for storing variables with layer name as key
        dic = {}
        entry = []
        remove_layer = False
        cut_squeeze1 = False
        cut_squeeze2 = False
        lay_gamma = 1

        for v_num, var in enumerate(weights):
            # If factor is below threshold(0.1), set layer to be removed
            if 'gamma_layer:0' in var.name:
                lay_gamma = sess.run(var)
                print(lay_gamma)
                self.lay_gammas = np.append(self.lay_gammas, lay_gamma)
                if lay_gamma < threshold:
                    self.dic_layers_pruned[var.name[:-13]] = True
                    remove_layer = True
                    if 'expand3x3' in var.name:
                        print('prune 3x3!')
                        cut_squeeze1 = True
                    elif 'expand1x1' in var.name:
                        print('prune 1x1!')
                        cut_squeeze2 = True
                    else:
                        print('no cutting error')
                        pdb.set_trace()
                else:
                    self.dic_layers_pruned[var.name[:-13]] = False

            # Remove half of connected squeeze layer if expand has been removed
            if 'kernels:0' in var.name:
                kernel = sess.run(var)
                if 'squeeze1x1/kernels' in var.name and cut_squeeze1:
                    kernel = kernel[:, :, :int(kernel.shape[2]/2), :]
                    cut_squeeze1 = False
                elif 'squeeze1x1/kernels' in var.name and cut_squeeze2:
                    kernel = kernel[:, :, int(kernel.shape[2]/2):, :]
                    cut_squeeze2 = False
                elif 'conv12/kernels' in var.name and cut_squeeze1:
                    kernel = kernel[:, :, :int(kernel.shape[2]/2), :]
                    cut_squeeze1 = False
                elif 'conv12/kernels' in var.name and cut_squeeze2:
                    kernel = kernel[:, :, int(kernel.shape[2]/2):, :]
                    cut_squeeze2 = False

                #multiply layer with factor gamma
                kernel_pruned = kernel*lay_gamma
                lay_gamma = 1

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
                if not remove_layer:
                    # Store all variables in dictionary
                    entry = [kernel_pruned, biases]
                    dic[var.name[:-9]] = entry
                remove_layer = False
                entry = []

        joblib.dump(dic,'SqueezeDetPruned1.pkl',compress=False )

    # Restore checkpoint
    def restore_checkpoint(self, sess):
        saver = tf.train.Saver(tf.global_variables())
        if os.path.isfile(self.checkpoint_dir + '.meta'):
          saver.restore(sess, self.checkpoint_dir)


Pruner = l1_model_pruner(0.1)
