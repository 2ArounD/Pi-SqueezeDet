import os
import tensorflow as tf
from config import kitti_squeezeDetPlus_config
from nets import SqueezeDetPlusPruneLayer, SqueezeDetPlusPruneFilter, SqueezeDetPlusPruneFilterShape


weights_path = '/path/to/weights/SqueezeDetPrunedFilterShape.pkl' # This is used to create smaller graph as well
export_dir = os.path.join('/path/to/desired/export_dir/export_dir', '0')
print(export_dir)


graph = tf.Graph()
with tf.compat.v1.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    mc = kitti_squeezeDetPlus_config()
    mc.IS_TRAINING = False
    mc.BATCH_SIZE = 1
    mc.LITE_MODE = True
    mc.IS_PRUNING = False
    mc.PRETRAINED_MODEL_PATH = weights_path
    model = SqueezeDetPlusPruneFilter(mc)  #Set this to the corresponding pruned structures of the model

    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    init_new_vars_op = tf.initialize_variables(uninitialized_vars)
    sess.run(init_new_vars_op)

    input_tensor_info = tf.saved_model.build_tensor_info(graph.get_tensor_by_name('image_input:0'))
    output_tensor_bbox_info = tf.saved_model.build_tensor_info(graph.get_tensor_by_name('bbox/trimming/bbox:0'))
    output_tensor_class_info = tf.saved_model.build_tensor_info(graph.get_tensor_by_name('probability/class_idx:0'))
    output_tensor_conf_info = tf.saved_model.build_tensor_info(graph.get_tensor_by_name('probability/score:0'))

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={'image': input_tensor_info},
                            outputs={'bbox': output_tensor_bbox_info,
                                     'class': output_tensor_class_info,
                                     'conf': output_tensor_conf_info},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # Export graph to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                         'predict_images': prediction_signature},
                                         clear_devices=True,
                                         strip_default_attrs=True)
    builder.save()
    builder.save(as_text=True)
