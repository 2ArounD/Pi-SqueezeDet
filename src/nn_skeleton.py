# Original author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Adapted by: A.Y.A. Jonker (arnoudjonker@gmail.com) 06/03/2020

"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
import numpy as np
import tensorflow as tf

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Add losses to summary for evalutaions during training and pruning
def _add_loss_summaries(total_loss):
  losses = tf.get_collection('losses')
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name, l)

# Function to create a Tensorflow variable
def _variable_on_device(name, shape, initializer, trainable=True):
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

# Function to create a Tensorflow variable with weight decay
def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

# Base class of the Squeezedet models
class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc):
    self.mc = mc
    # Variable for dropout
    self.keep_prob = 0.5 if mc.IS_TRAINING else 1.0

    # image batch input
    self.ph_image_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_input'
    )
    # A tensor where an element is 1 if the corresponding box is "responsible"
    # for detection an object and 0 otherwise.
    self.ph_input_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 1], name='box_mask')
    # Tensor used to represent bounding box deltas.
    self.ph_box_delta_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_delta_input')
    # Tensor used to represent bounding box coordinates.
    self.ph_box_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_input')
    # Tensor used to represent labels
    self.ph_labels = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES], name='labels')

    # IOU between predicted anchors with ground-truth boxes
    self.ious = tf.Variable(
      initial_value=np.zeros((mc.BATCH_SIZE, mc.ANCHORS)), trainable=False,
      name='iou', dtype=tf.float32
    )

    if not self.mc.LITE_MODE:
      # Queue for feeding data during training and pruning
      self.FIFOQueue = tf.FIFOQueue(
          capacity=mc.QUEUE_CAPACITY,
          dtypes=[tf.float32, tf.float32, tf.float32,
                  tf.float32, tf.float32],
          shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
                  [mc.ANCHORS, 1],
                  [mc.ANCHORS, 4],
                  [mc.ANCHORS, 4],
                  [mc.ANCHORS, mc.CLASSES]],
      )

      # Queue operation
      self.enqueue_op = self.FIFOQueue.enqueue_many(
          [self.ph_image_input, self.ph_input_mask,
           self.ph_box_delta_input, self.ph_box_input, self.ph_labels]
      )

      #
      self.image_input, self.input_mask, self.box_delta_input, \
          self.box_input, self.labels = tf.train.batch(
              self.FIFOQueue.dequeue(), batch_size=mc.BATCH_SIZE,
              capacity=mc.QUEUE_CAPACITY)

    if self.mc.LITE_MODE:
      self.image_input, self.input_mask, self.box_delta_input, \
        self.box_input, self.labels = [self.ph_image_input, self.ph_input_mask,
         self.ph_box_delta_input, self.ph_box_input, self.ph_labels]

    # model parameters for keeping track of parameters in model
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size
    # flop counter for keeping track of flops in model
    self.flop_counter = [] # array of tuple of layer name, flop number
    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.IMAGE_WIDTH*mc.IMAGE_HEIGHT*3))

  # Function to create network graph, to be defined by model class
  def _add_forward_graph(self):
    raise NotImplementedError

  # Function to create interpertation graph
  def _add_interpretation_graph(self):
    mc = self.mc

    with tf.variable_scope('interpret_output') as scope:
      preds = self.preds

      # probability
      num_class_probs = mc.ANCHOR_PER_GRID*mc.CLASSES
      self.pred_class_probs = tf.reshape(
          tf.nn.softmax(
              tf.reshape(
                  preds[:, :, :, :num_class_probs],
                  [-1, mc.CLASSES]
              )
          ),
          [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
          name='pred_class_probs'
      )

      # confidence
      num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
      self.pred_conf = tf.sigmoid(
          tf.reshape(
              preds[:, :, :, num_class_probs:num_confidence_scores],
              [mc.BATCH_SIZE, mc.ANCHORS]
          ),
          name='pred_confidence_score'
      )

      # bbox_delta
      self.pred_box_delta = tf.reshape(
          preds[:, :, :, num_confidence_scores:],
          [mc.BATCH_SIZE, mc.ANCHORS, 4],
          name='bbox_delta'
      )

      # number of object. Used to normalize bbox and classification loss
      self.num_objects = tf.reduce_sum(self.input_mask, name='num_objects')

    with tf.variable_scope('bbox') as scope:
      with tf.variable_scope('stretching'):
        delta_x, delta_y, delta_w, delta_h = tf.unstack(
            self.pred_box_delta, axis=2)

        anchor_x = mc.ANCHOR_BOX[:, 0]
        anchor_y = mc.ANCHOR_BOX[:, 1]
        anchor_w = mc.ANCHOR_BOX[:, 2]
        anchor_h = mc.ANCHOR_BOX[:, 3]

        box_center_x = tf.identity(
            anchor_x + delta_x * anchor_w, name='bbox_cx')
        box_center_y = tf.identity(
            anchor_y + delta_y * anchor_h, name='bbox_cy')
        box_width = tf.identity(
            anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH),
            name='bbox_width')
        box_height = tf.identity(
            anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH),
            name='bbox_height')

        self._activation_summary(delta_x, 'delta_x')
        self._activation_summary(delta_y, 'delta_y')
        self._activation_summary(delta_w, 'delta_w')
        self._activation_summary(delta_h, 'delta_h')

        self._activation_summary(box_center_x, 'bbox_cx')
        self._activation_summary(box_center_y, 'bbox_cy')
        self._activation_summary(box_width, 'bbox_width')
        self._activation_summary(box_height, 'bbox_height')

      with tf.variable_scope('trimming'):
        xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
            [box_center_x, box_center_y, box_width, box_height])

        # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
        # pixels. Same for y.
        xmins = tf.minimum(
            tf.maximum(0.0, xmins), mc.IMAGE_WIDTH-1.0, name='bbox_xmin')
        self._activation_summary(xmins, 'box_xmin')

        ymins = tf.minimum(
            tf.maximum(0.0, ymins), mc.IMAGE_HEIGHT-1.0, name='bbox_ymin')
        self._activation_summary(ymins, 'box_ymin')

        xmaxs = tf.maximum(
            tf.minimum(mc.IMAGE_WIDTH-1.0, xmaxs), 0.0, name='bbox_xmax')
        self._activation_summary(xmaxs, 'box_xmax')

        ymaxs = tf.maximum(
            tf.minimum(mc.IMAGE_HEIGHT-1.0, ymaxs), 0.0, name='bbox_ymax')
        self._activation_summary(ymaxs, 'box_ymax')

        self.det_boxes = tf.transpose(
            tf.stack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
            (1, 2, 0), name='bbox'
        )

    with tf.variable_scope('IOU'):
      def _tensor_iou(box1, box2):
        with tf.variable_scope('intersection'):
          xmin = tf.maximum(box1[0], box2[0], name='xmin')
          ymin = tf.maximum(box1[1], box2[1], name='ymin')
          xmax = tf.minimum(box1[2], box2[2], name='xmax')
          ymax = tf.minimum(box1[3], box2[3], name='ymax')

          w = tf.maximum(0.0, xmax-xmin, name='inter_w')
          h = tf.maximum(0.0, ymax-ymin, name='inter_h')
          intersection = tf.multiply(w, h, name='intersection')

        with tf.variable_scope('union'):
          w1 = tf.subtract(box1[2], box1[0], name='w1')
          h1 = tf.subtract(box1[3], box1[1], name='h1')
          w2 = tf.subtract(box2[2], box2[0], name='w2')
          h2 = tf.subtract(box2[3], box2[1], name='h2')

          union = w1*h1 + w2*h2 - intersection

        return intersection/(union+mc.EPSILON) \
            * tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

      self.ious = self.ious.assign(
          _tensor_iou(
              util.bbox_transform(tf.unstack(self.det_boxes, axis=2)),
              util.bbox_transform(tf.unstack(self.box_input, axis=2))
          )
      )
      self._activation_summary(self.ious, 'conf_score')

    with tf.variable_scope('probability') as scope:
      self._activation_summary(self.pred_class_probs, 'class_probs')

      probs = tf.multiply(
          self.pred_class_probs,
          tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          name='final_class_prob'
      )

      self._activation_summary(probs, 'final_class_prob')

      self.det_probs = tf.reduce_max(probs, 2, name='score')
      self.det_class = tf.argmax(probs, 2, name='class_idx')

  # Function to create graph of loss from model
  def _add_loss_graph(self):
    mc = self.mc

    with tf.variable_scope('class_regression') as scope:
      # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
      # add a small value into log to prevent blowing up
      self.class_loss = tf.truediv(
          tf.reduce_sum(
              (self.labels*(-tf.log(self.pred_class_probs+mc.EPSILON))
               + (1-self.labels)*(-tf.log(1-self.pred_class_probs+mc.EPSILON)))
              * self.input_mask * mc.LOSS_COEF_CLASS),
          self.num_objects,
          name='class_loss'
      )
      tf.add_to_collection('losses', self.class_loss)

    with tf.variable_scope('confidence_score_regression') as scope:
      input_mask = tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
      self.conf_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square((self.ious - self.pred_conf))
              * (input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects
                 +(1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects)),
              reduction_indices=[1]
          ),
          name='confidence_loss'
      )
      tf.add_to_collection('losses', self.conf_loss)
      tf.summary.scalar('mean iou', tf.reduce_sum(self.ious)/self.num_objects)

    with tf.variable_scope('bounding_box_regression') as scope:
      self.bbox_loss = tf.truediv(
          tf.reduce_sum(
              mc.LOSS_COEF_BBOX * tf.square(
                  self.input_mask*(self.pred_box_delta-self.box_delta_input))),
          self.num_objects,
          name='bbox_loss'
      )
      tf.add_to_collection('losses', self.bbox_loss)

    # Added functionality to add for pruning structures of the model
    if self.mc.IS_PRUNING:
      with tf.variable_scope('l1_regularization') as scope:

        gammas = [par for par in self.model_params if 'gamma' in par.name]
        if mc.IS_PRUNING:
          len_gammas=0
          for g in gammas:
            len_gammas = len_gammas + int(g.shape[0])

          lamb=4.5/len_gammas
        else:
          lamb = 0.0

        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=lamb, scope=None)
        self.gamma_loss = tf.contrib.layers.apply_regularization(l1_regularizer, gammas)

        tf.add_to_collection('losses', self.gamma_loss)
    else:
      # with tf.variable_scope('l1_regularization') as scope:
        #Create dummy loss of 0 when not pruning
        dummy = _variable_on_device('dummy', [1], tf.constant_initializer(0.0),  False)
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0, scope=None)
        self.gamma_loss = tf.contrib.layers.apply_regularization(l1_regularizer, [dummy])

        tf.add_to_collection('losses', self.gamma_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  # Function to create graph for training operations
  def _add_train_graph(self):
    mc = self.mc

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    _add_loss_summaries(self.loss)

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
    grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())
    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads_vars:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')

  # Function to add graph for vizualisation during training
  def _add_viz_graph(self):
    mc = self.mc
    self.image_to_show = tf.placeholder(
        tf.float32, [None, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_to_show'
    )
    self.viz_op = tf.summary.image('sample_detection_results',
        self.image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)

  # Convolutional layer with pruning functionality included
  def _conv_layer(
      self, inputs, conv_param_name, filters, size, stride,
      slc=[[0, 0], [0, 0]], padding='SAME', pruning=False, prune_struct='', relu=True,
      stddev=0.001):

    mc = self.mc

    with tf.variable_scope(conv_param_name) as scope:
      channels = inputs.get_shape()[3]

      # Create variables for pruning
      if pruning and prune_struct=='filter':
        gamma_val = tf.constant_initializer(1.0)
        gamma_filter = _variable_on_device('gamma_filter', [filters], gamma_val,
                                    trainable=(pruning))
        self.model_params += [gamma_filter]

      # Create variables for pruning
      if pruning and prune_struct=='f_shape' and np.size(size)>1:
        mask_row_val = tf.initializers.identity()
        mask_col_val = tf.initializers.identity()
        identity_row_val = tf.initializers.identity()
        identity_col_val = tf.initializers.identity()

        gamma_row = _variable_on_device('gamma_row', [size[1], size[1]],
                                        mask_row_val, trainable=(pruning))

        gamma_col = _variable_on_device('gamma_col', [size[0], size[0]],
                                        mask_col_val, trainable=(pruning))
        mask_gamma_row = _variable_on_device('id_fil_row', [size[1], size[1]],
                                             identity_row_val, trainable=False)
        mask_gamma_col = _variable_on_device('id_fil_col', [size[0], size[0]],
                                             identity_col_val, trainable=False)
        self.model_params += [gamma_row]
        self.model_params += [gamma_col]

      # Create variables for pruning
      if pruning and prune_struct=='layer':
        layer_gamma_val  = tf.constant_initializer(1.0)
        layer_gamma = _variable_on_device('gamma_layer', [1], layer_gamma_val,
                                    trainable=pruning)
        self.model_params += [layer_gamma]

      # Create kernels and biases
      if mc.LOAD_PRETRAINED_MODEL:
        cw = self.model_weights
        kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
        bias_val = cw[conv_param_name][1]
      else:
        kernel_val = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_val = tf.constant_initializer(0.0)

      kernel_init = tf.constant(kernel_val , dtype=tf.float32)
      bias_init = tf.constant(bias_val, dtype=tf.float32)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not pruning))

      biases = _variable_on_device('biases', [filters], bias_init,
                                trainable=(not pruning))

      self.model_params += [kernel, biases]

      # Add gammas to filters when pruning filters
      if pruning and prune_struct=='filter':
        kernel = tf.multiply(gamma_filter, kernel)

      # Add gammas to filter height and width when pruning those
      if pruning and prune_struct=='f_shape' and np.size(size)>1:
        mask_row_filtered = tf.multiply(mask_gamma_row, gamma_row)
        mask_col_filtered = tf.multiply(mask_gamma_col, gamma_col)
        kernel = tf.transpose(tf.matmul(tf.expand_dims(tf.expand_dims(
                              mask_row_filtered, 0), 0), tf.transpose(kernel),
                              name = 'maskmultirow'))
        kernel = tf.transpose(tf.matmul(tf.transpose(kernel),
                              tf.expand_dims(tf.expand_dims(
                                mask_col_filtered, 0), 0),
                              name = 'maskmulticol'))

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv = tf.nn.bias_add(conv, biases, name='bias_add')

      # Add gamma to complete layer if pruning layers
      if pruning and prune_struct=='layer':
        conv = tf.multiply(conv, layer_gamma)

      out_shape = conv.get_shape().as_list()

      if np.size(size)>1:
        self.model_size_counter.append(
            (conv_param_name, (1+size[0]*size[1]*int(channels))*filters))
        num_flops = \
        (1+2*int(channels)*size[0]*size[1])*filters*out_shape[1]*out_shape[2]
      else:
        self.model_size_counter.append(
            (conv_param_name, (1+size*size*int(channels))*filters))
        num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]

      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((conv_param_name, num_flops))

      self.activation_counter.append(
          (conv_param_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      if relu:
        return tf.nn.relu(conv)
      else:
        return conv

  # Pooling layer for networks
  def _pooling_layer(
      self, layer_name, inputs, size, stride, padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs,
                            ksize=[1, size, size, 1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out

  # Filter the object box prediction by probability
  def filter_prediction(self, boxes, probs, cls_idx):
    mc = self.mc

    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
    else:
      filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(mc.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx

  # Create summaries of layers activations
  def _activation_summary(self, x, layer_name):

    with tf.variable_scope('activation_summary') as scope:
      tf.summary.histogram(
          'activation_summary/'+layer_name, x)
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/min', tf.reduce_min(x))
