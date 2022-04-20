from ast import For
from typing import Any
import numpy as np
from core.config import cfg
from core.utils import read_class_names
import csv

def countObjects(data: list, byClass: bool = True,  allowedClasses: list = list(read_class_names(cfg.YOLO.CLASSES).values())) -> dict:
    boxes, scores, classes, numObjects = data
    counts = dict()

    # byClass = True => count objects per class
    if byClass:
        classNames = read_class_names(cfg.YOLO.CLASSES)
        for i in range(numObjects):
            classIndex = int(classes[i])
            className = classNames[classIndex]
            if className in allowedClasses:
                counts[className] = counts.get(className, 0) + 1
            else:
                continue

    # count all objects
    else:
        counts['all objects'] = numObjects
    
    return counts

# def generateCsv(time: Any, typeDetected: Any, numDetections: Any):
#     with open('resultInfo.csv', 'w', newline='') as csvfile:
#         fieldNames = ['Time', 'NumberObject', 'TypeObject']
#         writer = csv.DictWriter(csvfile, fieldnames = fieldNames)

#         writer.writeheader()
#         writer.writerow({
#             'Time': time,
#             'NumberObject': numDetections,
#             'TypeObject': typeDetected,
#         })
#     del writer
#     csvfile.close()

def generateCsv(info):
    with open('resultInfo.csv', 'w', newline='') as csvfile:
        fieldNames = ['Time', 'NumberObject', 'TypeObject']
        writer = csv.DictWriter(csvfile, fieldnames = fieldNames)
        writer.writeheader()
        for i in info:
            writer.writerow({
                'Time': i[0], 
                'NumberObject': i[2], 
                'TypeObject': i[1]
            })
    del writer
    csvfile.close()
