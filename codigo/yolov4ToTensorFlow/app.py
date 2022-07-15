# Flask dependecies
import csv
from datetime import datetime
from http.client import responses
from flask import Flask, request, redirect, url_for, render_template, abort, Response

import tensorflow as tf
from core.functions import count_objects_img
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.config import cfg
from core.yolov4 import decode, filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
import os
import json
import requests

# Basic configuration
framework = 'tf'
weights = './checkpoints/yolov4-416'
size = 416
tiny = False
model = 'yolov4'
improve_quality = False
output_path = './static/detections/'
iou = 0.45
score = 0.25
allow_classes = list(utils.read_class_names(cfg.YOLO.CLASSES).values())

class Flag:
    tiny = tiny
    model = model
    
    
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = Flag
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = size

# Load model
if framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=weights)
else:
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])

# Initialize Flask app     
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
print("app loaded")

@app.route('/')
def home():
    return render_template('./index.html')

# Returns the image with detections on it
@app.route('/image/detections', methods=['POST'])
def get_image_detections():
    imgs = request.files.getlist('images')
    img_path_list = []
    for img in imgs:
        img_name = img.filename
        img_path_list.append("./temp/" + img_name)
        img.save(os.path.join(os.getcwd(), "temp/", img_name))

    response = []
    for count, img_path in enumerate(img_path_list):
        responses = []
        try:
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            img_data = cv2.resize(original_img, (input_size, input_size))
            img_data = img_data / 255.
        except cv2.error:
            # Remove temporary images
            for name in img_path_list:
                os.remove(name)
            abort(404, "it is not an image file or image file is an unsupported format. try jpg or png")
        except Exception as e:
            # Remove temporary images
            for name in img_path_list:
                os.remove(name)
            print(e.__class__)
            print(e)
            abort(500)

        imgs_data = []

        for _ in range(1):
            imgs_data.append(img_data)
        imgs_data = np.asarray(imgs_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], imgs_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

            if improve_quality == True:
                bbox_tensors = []
                prob_tensors = []
                for i, _ in enumerate(pred):
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

            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            t1 = time.time()
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(imgs_data)
            pred_bbox = infer(batch_data)
            for _, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            t2 = time.time()
            print('time: {}'.format(t2 - t1))

        t1 = time.time()
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        t2 = time.time()
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        print('time: {}'.format(t2 - t1))
        for i in range(valid_detections[0]):
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100)),
                "box": np.array(boxes[0][i]).tolist()
            })
        response.append({
            "image": img_path_list[count][7:],
            "detections": responses
        })
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        allowed_classes = allow_classes

        counted_classes = count_objects_img(pred_bbox, by_class=True, allowed_classes=allowed_classes)
        image, pos = utils.draw_bbox_img(original_img, pred_bbox, allowed_classes=allowed_classes)
        results = []

        for key, value in counted_classes.items():
            for k, v in pos.items():
                if key == k:
                    results.append([datetime.now(), key, value, v[:]])
                
        name_csv = output_path + img_name[0:len(img_name)-4] + '.csv'
        with open(name_csv, 'w', newline='') as csvfile:
            field_names = ['Time', 'NumberObject', 'TypeObject', 'Positions']
            writer = csv.DictWriter(csvfile, fieldnames = field_names)
            writer.writeheader()
            for i in results:
                writer.writerow({
                    'Time': i[0], 
                    'NumberObject': i[2], 
                    'TypeObject': i[1],
                    'Positions': str(i[3]),
                })
        del writer
        csvfile.close()

        image = Image.fromarray(image.astype(np.uint8))

        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + img_name[0:len(img_name)-4] + '.png', image)

    # Remove temporary images
    for name in img_path_list:
        os.remove(name)
    try:
        return Response(response=json.dumps({"response": response}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)