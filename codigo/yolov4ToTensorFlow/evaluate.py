from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from core.config import cfg
from PIL import Image 

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, trt)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('classes', '', 'path to input classes in positions')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    INPUT_SIZE = FLAGS.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    if FLAGS.classes:
        CLASSES = utils.read_class_names(FLAGS.classes)
        NUM_CLASS = len(CLASSES)
    else:
        CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    predicted_dir_path = './mAP/predicted'
    data_path = './mAP/license_plate'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    num_lines = sum(1 for line in open(FLAGS.annotation_path))

    if FLAGS.annotation_path:
        fich = FLAGS.annotation_path
    else:
        fich = cfg.TEST.ANNOT_PATH 

    with open(fich, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]

            image = cv2.imread(image_path)
            h, w, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    class_name = class_name.replace(" ", "-")
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
                    bbox_mess_poss = list(bbox_mess.split(' '))
                    bbox_mess_poss.pop(0)
                    if not bbox_mess_poss[0].isdigit():
                        bbox_mess_poss.pop(0)

                    c1, c2 = (int(bbox_mess_poss[0]), int(bbox_mess_poss[1])), (int(bbox_mess_poss[2]), int(bbox_mess_poss[3].replace('\n', '')))
                    cv2.rectangle(image, c1, c2, (0, 255, 0), 2)
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
            else:
                batch_data = tf.constant(image_data)
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
            boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            
            gt = pd.read_csv(ground_truth_path, sep=' ', names=['file', 'x1', 'y1', 'x2', 'y2'])
            gt = gt.drop(['file'], axis=1)

            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape
                for i in range(valid_detections[0]):
                    iou = -1
                    if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                    coor = boxes[0][i]
                    
                    coor[0] = int(coor[0] * image_h)
                    coor[2] = int(coor[2] * image_h)
                    coor[1] = int(coor[1] * image_w)
                    coor[3] = int(coor[3] * image_w)

                    score = scores[0][i]
                    class_ind = int(classes[0][i])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
                    bbox_pred = list(bbox_mess.split(' '))
                    bbox_pred.pop(0)
                    bbox_pred.pop(0)
                    # pri8nt(int(float(bbox_pred[0])))
                    for k in gt.index: 
                        gt_k = gt.iloc[k].to_list()
                        if iou < utils.bb_intersection_over_union(gt_k, bbox_pred):
                            iou = utils.bb_intersection_over_union(gt_k, bbox_pred)

                    c1, c2 = (int(float(bbox_pred[0])), int(float(bbox_pred[1]))), (int(float(bbox_pred[2])), int(float(bbox_pred[3].replace('\n', ''))))
                    cv2.rectangle(image, c1, c2, (0, 0, 255), 2)
                    cv2.putText(image, "IoU: {:.4f}".format(iou),  (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (238, 255, 3), 2, lineType=cv2.LINE_AA)
            print(num, num_lines)

            data_predict_result_path = os.path.join(data_path, 'data' + str(num) + '.txt')
            with open(data_predict_result_path, 'w') as d:
                textIoU = f"IoU {iou}"
                d.write(textIoU + ' ' + bbox_mess)

            image = Image.fromarray(image.astype(np.uint8))
            # image.show()
            image_path = os.path.join(predicted_dir_path, str(num) + '.png')
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite((image_path), image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


