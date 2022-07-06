import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import decode, filter_boxes
from core.functions import count_objects_img, generate_csv
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from datetime import datetime

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/images/image.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/images/result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_list('allowed_classes', list(utils.read_class_names(cfg.YOLO.CLASSES).values()), 'list of allowed classes')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image
    results = []

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for _ in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        bbox_tensors = []
        prob_tensors = []
        for i, fm in enumerate(pred):
            if i == 0:
                output_tensors = decode(pred[2], input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            elif i == 1:
                output_tensors = decode(pred[0], input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            else:
                output_tensors = decode(pred[1], input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
        pred_bbox = tf.concat(bbox_tensors, axis=1)
        pred_prob = tf.concat(prob_tensors, axis=1)
        pred = (pred_bbox, pred_prob)

        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    counted_classes = count_objects_img(pred_bbox, by_class=True, allowed_classes=FLAGS.allowed_classes)
    image, registro_pos = utils.draw_bbox_img(original_image, pred_bbox, allowed_classes=FLAGS.allowed_classes)
    for key, value in counted_classes.items():
        for k, v in registro_pos.items():
            if key == k:
                results.append([datetime.now(), key, value, v[:]])
    generate_csv(results, FLAGS.image)

    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)

    print("Image processing complete")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
