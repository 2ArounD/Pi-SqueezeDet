import os
import tensorflow as tf


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    export_dir = os.path.join('/home/arnoud/Documents/TU/Afstuderen/Code/github/output/export_dir', '0')

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir, signature_key='predict_images')
    tflite_model = converter.convert()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    open("/home/arnoud/Documents/TU/Afstuderen/Code/github/output/tflite_models/model_layer.tflite", "wb").write(tflite_model)




