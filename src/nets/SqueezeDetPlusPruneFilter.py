
"""SqueezeDet+ model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton
import pdb

class SqueezeDetPlusPruneFilter(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)
      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc

    assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
        'Cannot find pretrained model at the given path:' \
        '  {}'.format(mc.PRETRAINED_MODEL_PATH)
    self.model_weights = joblib.load(mc.PRETRAINED_MODEL_PATH)

    conv1 = self._conv_layer(self.image_input,
        'conv1',
        filters=self.flt('conv1'), size=7, stride=2,
        padding='VALID', pruning=mc.IS_PRUNING, prune_struct='filter')

    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='VALID')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=96, e1x1=64, e3x3=64, pruning=mc.IS_PRUNING)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=96, e1x1=64, e3x3=64, pruning=mc.IS_PRUNING)
    fire4 = self._fire_layer(
        'fire4', fire3, s1x1=192, e1x1=128, e3x3=128, pruning=mc.IS_PRUNING)
    pool4 = self._pooling_layer(
        'pool4', fire4, size=3, stride=2, padding='VALID')

    fire5 = self._fire_layer(
        'fire5', pool4, s1x1=192, e1x1=128, e3x3=128, pruning=mc.IS_PRUNING)
    fire6 = self._fire_layer(
        'fire6', fire5, s1x1=288, e1x1=192, e3x3=192, pruning=mc.IS_PRUNING)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=288, e1x1=192, e3x3=192, pruning=mc.IS_PRUNING)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=384, e1x1=256, e3x3=256, pruning=mc.IS_PRUNING)
    pool8 = self._pooling_layer(
        'pool8', fire8, size=3, stride=2, padding='VALID')

    fire9 = self._fire_layer(
        'fire9', pool8, s1x1=384, e1x1=256, e3x3=256, pruning=mc.IS_PRUNING)

    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=384, e1x1=256, e3x3=256, pruning=mc.IS_PRUNING)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=384, e1x1=256, e3x3=256, pruning=mc.IS_PRUNING)
    dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(dropout11,
        'conv12',
        filters=num_output, size=3, stride=1,
        padding='SAME', relu=False, stddev=0.0001,
        pruning=False)


  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      pruning=False, prune_struct='filter'):
    """Fire layer constructor"""

    sq1x1 = self._conv_layer(inputs,
        layer_name+'/squeeze1x1',
        filters=self.flt(layer_name+'/squeeze1x1'), size=1, stride=1,
        padding='SAME', stddev=stddev, pruning=pruning, prune_struct=prune_struct)
    ex1x1 = self._conv_layer(sq1x1,
        layer_name+'/expand1x1',
        filters=self.flt(layer_name+'/expand1x1'), size=1, stride=1,
        padding='SAME', stddev=stddev, pruning=pruning, prune_struct=prune_struct)
    ex3x3 = self._conv_layer(sq1x1,
        layer_name+'/expand3x3',
        filters=self.flt(layer_name+'/expand3x3'), size=3, stride=1,
        padding='SAME', stddev=stddev, pruning=pruning, prune_struct=prune_struct)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')


  def flt(self, layer_name):
    return self.model_weights[layer_name][0].shape[0]
