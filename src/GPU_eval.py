from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import pickle
from six.moves import xrange
import tensorflow as tf
from time import process_time
import numpy as np

from config import *
from dataset import kitti
from utils.util import bbox_transform, Timer
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', '../data/KITTI', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('eval_dir', '',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('pretrained_model_path', 'path/to/pretrained-paramteers/SqueezeDetPlus/SqueezeDetPlus.pkl',
                            """Used to initialize paramters and to derive structure of network! """)
tf.app.flags.DEFINE_string('checkpoint_path', '/',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                             """How often to check if new ckpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet+PruneFilter',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def eval_once(
    saver, ckpt_path, imdb, model, step, restore_checkpoint):

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    if restore_checkpoint:
      saver.restore(sess, ckpt_path)

    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    init_new_vars_op = tf.initialize_variables(uninitialized_vars)
    sess.run(init_new_vars_op)

    num_images = len(imdb.image_idx)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # Detection sequence, looping through all images
    _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer()}
    num_detection = 0.0
    process_t = np.array([])
    for i in xrange(num_images):
        _t['im_read'].tic()
        images, scales = imdb.read_image_batch(shuffle=False)
        _t['im_read'].toc()
        _t['im_detect'].tic()
        t_start=process_time() #Using process time to measure detection time
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:images})
        t_stop = process_time() #Using process time to measure detection time
        process_t = np.append(process_t, t_stop-t_start)
        _t['im_detect'].toc()
        _t['misc'].tic()
        for j in range(len(det_boxes)): # batch
            # rescale
            det_boxes[j, :, 0::2] /= scales[j][0]
            det_boxes[j, :, 1::2] /= scales[j][1]

            det_bbox, score, det_class = model.filter_prediction(
                det_boxes[j], det_probs[j], det_class[j])

            num_detection += len(det_bbox)
            for c, b, s in zip(det_class, det_bbox, score):
                all_boxes[c][i].append(bbox_transform(b) + [s])
            _t['misc'].toc()

    if not os.path.exists(FLAGS.eval_dir + "/" + step):
      os.mkdir(FLAGS.eval_dir + "/" + step)

    #Save all evaluation data
    pickle.dump(all_boxes, open(FLAGS.eval_dir + "/" + step + "/all_boxes.p", "wb"))
    pickle.dump(_t, open(FLAGS.eval_dir + "/" + step + "/_t.p", "wb"))
    pickle.dump(num_detection, open(FLAGS.eval_dir + "/" + step + "/num_detection.p", "wb"))
    pickle.dump(process_t, open(FLAGS.eval_dir + "/" + step + "/process_t.p", "wb"))

def evaluate():
  """Evaluate."""
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  if not os.path.exists(FLAGS.eval_dir):
    os.makedirs(FLAGS.eval_dir)

  with tf.Graph().as_default() as g:

    #Select model to evaluate
    if FLAGS.net == 'squeezeDet+PruneFilter':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.IS_TRAINING = False
      mc.IS_PRUNING = False
      mc.LOAD_PRETRAINED_MODEL = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlusPruneFilter(mc)
    elif FLAGS.net == 'squeezeDet+PruneFilterShape':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.IS_TRAINING = False
      mc.IS_PRUNING = False
      mc.LOAD_PRETRAINED_MODEL = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlusPruneFilterShape(mc)
    elif FLAGS.net == 'squeezeDet+PruneLayer':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.IS_TRAINING = False
      mc.IS_PRUNING = False
      mc.LOAD_PRETRAINED_MODEL = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlusPruneLayer(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    no_evaluation = True
    # Logic to load checkpoints if available
    for file in os.listdir(FLAGS.checkpoint_path):
      if file.endswith(".meta"):
        saver = tf.train.Saver(model.model_params)
        ckpt_path = FLAGS.checkpoint_path + "/" + file[:-5]
        step = file[11:-5]
        eval_once(saver, ckpt_path, imdb, model, step, True)
        no_evaluation = False

    if no_evaluation:
      saver = None
      eval_once(saver, '-', imdb, model, '0', False)

def main(argv=None):  #
  evaluate()

if __name__ == '__main__':
  tf.app.run()
