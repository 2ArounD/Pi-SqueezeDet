import numpy as np
from dataset import kitti
import tensorflow as tf
from config import *
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'path/to/checkpoint/1999', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('pretrained_model_path', '../SqueezeDetPrunedBN1.pkl',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('data_path', '../data/KITTI_2', """Root directory of data""")




with tf.Graph().as_default() as g:

    mc = kitti_squeezeDetPlus_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = True
    print(FLAGS.pretrained_model_path)
    mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # add summary ops and placeholders
    ap_names = []
    for cls in imdb.classes:
      ap_names.append(cls+'_easy')
      ap_names.append(cls+'_medium')
      ap_names.append(cls+'_hard')

    eval_summary_ops = []
    eval_summary_phs = {}
    for ap_name in ap_names:
      ph = tf.placeholder(tf.float32)
      eval_summary_phs['APs/'+ap_name] = ph
      eval_summary_ops.append(tf.summary.scalar('APs/'+ap_name, ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['APs/mAP'] = ph
    eval_summary_ops.append(tf.summary.scalar('APs/mAP', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/im_detect'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/im_detect', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/im_read'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/im_read', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/post_proc'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/post_proc', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['num_det_per_image'] = ph
    eval_summary_ops.append(tf.summary.scalar('num_det_per_image', ph))

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

with open(FLAGS.eval_dir + '/all_boxes.p', 'rb') as boxes_file:
    all_boxes = pickle.load(boxes_file)
with open(FLAGS.eval_dir + '/num_detection.p', 'rb') as num_file:
    num_detection = pickle.load(num_file)
with open(FLAGS.eval_dir + '/_t.p', 'rb') as t_file:
    _t = pickle.load(t_file)
num_images = len(imdb.image_idx)
global_step = '0'

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph = g) as sess:

    print ('Evaluating detections...')
    aps, ap_names = imdb.evaluate_detections(
        FLAGS.eval_dir, global_step, all_boxes)

    print ('Evaluation summary:')
    #print ('  Average number of detections per image: {}:'.format(
    #  num_detection/num_images))
    print ('  Timing:')
    print ('    im_read: {:.3f}s detect: {:.3f}s misc: {:.3f}s'.format(
      _t['im_read'].average_time, _t['im_detect'].average_time,
      _t['misc'].average_time))
    print ('  Average precisions:')

    feed_dict = {}
    for cls, ap in zip(ap_names, aps):
      feed_dict[eval_summary_phs['APs/'+cls]] = ap
      print ('    {}: {:.3f}'.format(cls, ap))

    # res_file = open ('./data/results/first_pruning.txt', 'a+')
    # res_file.write(FLAGS.eval_dir + '\n')
    # res_file.write(global_step +'\n')
    # res_file.write('    Mean average precision: {:.3f}'.format(np.mean(aps)) + '\n')

    print ('    Mean average precision: {:.3f}'.format(np.mean(aps)))
    feed_dict[eval_summary_phs['APs/mAP']] = np.mean(aps)
    feed_dict[eval_summary_phs['timing/im_detect']] = \
        _t['im_detect'].average_time
    feed_dict[eval_summary_phs['timing/im_read']] = \
        _t['im_read'].average_time
    feed_dict[eval_summary_phs['timing/post_proc']] = \
        _t['misc'].average_time
    feed_dict[eval_summary_phs['num_det_per_image']] = \
        num_detection/num_images

    print ('Analyzing detections...')
    stats, ims = imdb.do_detection_analysis_in_eval(
        FLAGS.eval_dir, global_step)

    eval_summary_str = sess.run(eval_summary_ops, feed_dict=feed_dict)
    for sum_str in eval_summary_str:
      summary_writer.add_summary(sum_str, global_step)
