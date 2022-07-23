# Flask dependecies
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import csv
from datetime import datetime
from http.client import responses
from flask import Flask, request, redirect, url_for, render_template, abort, Response
from deep_sort.tracker import Tracker
from core.functions import count_objects, count_objects_img
from deep_sort import nn_matching
from tools import generate_detections as gdet
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

@app.route('/image')
def image():
    return render_template('./image.html')

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


@app.route('/video')
def video():
    return render_template('./video.html')

@app.route('/video/detections', methods=['POST'])
def get_video_detections():
    videos = request.files.getlist('videos')
    video_path_list = []
    for video in videos:
        video_name = video.filename
        video_path_list.append("./temp/" + video_name)
        video.save(os.path.join(os.getcwd(), "temp/", video_name))

    response = []

    for count, video_path in enumerate(video_path_list):
        responses = []
        results = []
        try:
            vid = cv2.VideoCapture(video_path)
        except cv2.error:
            for name in video_path_list:
                os.remove(name)
            abort(404, "it is not a video file or video file is an unsupported format. try mp4")  

        out = None

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path + video_name[0:len(video_name)-4] + '.mp4', -1, fps, (width, height))

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            t1 = time.time()
            infer = saved_model_loaded.signatures['serving_default']
        
        frame_id = 0
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Video processing complete")
                    break
                raise ValueError("No image! Try with another video format")

            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(
                    output_details[i]['index']) for i in range(len(output_details))]
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
                batch_data = tf.constant(image_data)
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
            
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0],
                        valid_detections.numpy()[0]]

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
                "video": video_path_list[count][7:],
                "detections": responses
            })

            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            allowed_classes = allow_classes

            counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
            image, registro_pos = utils.draw_bbox_info(frame, pred_bbox, allowed_classes=allowed_classes)
            for key, value in counted_classes.items():
                for k, v in registro_pos.items():
                    if key == k:
                        results.append([datetime.now(), key, value, v[:]])
            name_csv = output_path + video_name[0:len(video_name)-4] + '.csv'
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

            result = np.asarray(image)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(result)

            frame_id += 1
        vid.release()

        cv2.destroyAllWindows()
    # Remove temporary images
    for name in video_path_list:
        os.remove(name)
    try:
        return Response(response=json.dumps({"response": response}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)

@app.route('/url')
def url():
    return render_template('./url.html')

@app.route('/image_url', methods=['POST'])
def get_image_detections_url():
    image_urls = request.values.getlist('images')
    raw_image_list = []
    if not isinstance(image_urls, list):
        abort(400, "can't find image list")
    image_names = []
    custom_headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    }
    for i, image_url in enumerate(image_urls):
        print(image_url)
        orig = image_url.index('https://')
        end = image_url.index('.es') if image_url.find('.es') != -1 else image_url.index('.com')
        print(image_url[orig:end])
        image_name = image_url[orig:end]
        image_name = image_name.replace('https://', '')
        image_names.append(image_name)
        try:
            resp = requests.get(image_url, headers=custom_headers)
            img_raw = np.asarray(bytearray(resp.content), dtype="uint8")
            img_raw = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
        except cv2.error:
            abort(404, "it is not image url or that image is an unsupported format. try jpg or png")
        except requests.exceptions.MissingSchema:
            abort(400, "it is not url form")
        except Exception as e:
            print(e.__class__)
            print(e)
            abort(500)
        raw_image_list.append(img_raw)

    # create list for final response
    response = []
    # loop through images in list and run Yolov4 model on each
    for count, raw_image in enumerate(raw_image_list):
        # create list of responses for current image
        responses = []

        original_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
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
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
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

        if os.path.exists(output_path + image_name + '.png'):
            numb = 1
            while True:
                new_image_name = image_name + '_' + str(numb)
                if os.path.exists(output_path + new_image_name + '.png'):
                    numb += 1 
                else:
                    break
            image_name = new_image_name

        for i in range(valid_detections[0]):
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100)),
                "box": np.array(boxes[0][i]).tolist()
            })
        response.append({
            "image": image_name + '.png',
            "detections": responses
        })
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = allow_classes

        counted_classes = count_objects_img(pred_bbox, by_class=True, allowed_classes=allowed_classes)
        image, pos = utils.draw_bbox_img(original_image, pred_bbox, allowed_classes=allowed_classes)
        results = []

        for key, value in counted_classes.items():
            for k, v in pos.items():
                if key == k:
                    results.append([datetime.now(), key, value, v[:]])
        
        name_csv = output_path + image_name + '.csv'
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
        
        cv2.imwrite(output_path + image_name + '.png', image)

    try:
        return Response(response=json.dumps({"response": response}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)
