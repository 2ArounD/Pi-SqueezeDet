import os
import tensorflow as tf
from dataset import kitti
from config import kitti_squeezeDetPlus_config


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    export_dir = os.path.join('/path/to/export_dir', '0')

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir, signature_key='predict_images')
    tflite_model = converter.convert()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    mc = kitti_squeezeDetPlus_config()
    mc.IS_TRAINING = False
    mc.BATCH_SIZE = 1
    imdb = kitti('val', '../data/KITTI', mc)

    open("path/to/lite_model/SqueezeDetFull1.tflite", "wb").write(tflite_model)




