from core.config import cfg
from core.utils import read_class_names
import csv

def count_objects(data: list, by_class: bool = True,  allowed_classes: list = list(read_class_names(cfg.YOLO.CLASSES).values())) -> dict:
    boxes, scores, classes, numObjects = data
    counts = dict()

    # byClass = True => count objects per class
    if by_class:
        classNames = read_class_names(cfg.YOLO.CLASSES)
        for i in range(numObjects):
            classIndex = int(classes[i])
            className = classNames[classIndex]
            if className in allowed_classes:
                counts[className] = counts.get(className, 0) + 1
            else:
                continue

    # count all objects
    else:
        counts['all objects'] = numObjects
    
    return counts


def count_objects_img(data: list, by_class: bool = True,  allowed_classes: list = list(read_class_names(cfg.YOLO.CLASSES).values())) -> dict:
    boxes, scores, classes, numObjects = data
    counts = dict()
    classes = classes[0]
    numObjects = numObjects[0]

    # by_class = True => count objects per class
    if by_class:
        classNames = read_class_names(cfg.YOLO.CLASSES)
        for i in range(numObjects):
            classIndex = int(classes[i])
            className = classNames[classIndex]
            if className in allowed_classes:
                counts[className] = counts.get(className, 0) + 1
            else:
                continue

    # count all objects
    else:
        counts['all objects'] = numObjects
    return counts


def generate_csv(info: list, name_csv: str):
    name_csv = name_csv.replace('.mp4', '.csv')
    name_csv = name_csv.replace('.png', '.csv')
    name_csv = name_csv.replace('.jpg', '.csv')

    with open(name_csv, 'w', newline='') as csvfile:
        fieldNames = ['Time', 'NumberObject', 'TypeObject', 'Positions']
        writer = csv.DictWriter(csvfile, fieldnames = fieldNames)
        writer.writeheader()
        for i in info:
            writer.writerow({
                'Time': i[0], 
                'NumberObject': i[2], 
                'TypeObject': i[1],
                'Positions': str(i[3]),
            })
    del writer
    csvfile.close()
