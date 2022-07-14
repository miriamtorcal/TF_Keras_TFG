# Flask dependecies
from flask import Flask, request, redirect, url_for, render_template, abort
from flask_wtf.csrf import CSRFProtect

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
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
output_path = './detections/'
iou = 0.45
score = 0.25

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
csrf = CSRFProtect()
csrf.init_app(app)
app.config['SECRET_KEY'] = '5710c8ae51a4b5af97be6534caef90e4bb9bdcb3380af008f90b23a5d1616bf319bc298105da20ff'

print("app loaded")

@app.route('/')
def home():
    return render_template('./index.html')