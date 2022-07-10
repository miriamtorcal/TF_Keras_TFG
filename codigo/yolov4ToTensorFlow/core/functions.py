from core.config import cfg
from core.utils import read_class_names
import csv

def count_objects(data: list, by_class: bool = True,  allowed_classes: list = list(read_class_names(cfg.YOLO.CLASSES).values())) -> dict:
    boxes, scores, classes, num_objects = data
    counts = dict()

    # byClass = True => count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)
        for i in range(num_objects):
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # count all objects
    else:
        counts['all objects'] = num_objects
    
    return counts


def count_objects_img(data: list, by_class: bool = True,  allowed_classes: list = list(read_class_names(cfg.YOLO.CLASSES).values())) -> dict:
    boxes, scores, classes, num_objects = data
    counts = dict()
    classes = classes[0]
    num_objects = num_objects[0]

    # by_class = True => count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)
        for i in range(num_objects):
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # count all objects
    else:
        counts['all objects'] = num_objects
    return counts


def generate_csv(info: list, name_csv: str):
    name_csv = name_csv.replace('.mp4', '.csv')
    name_csv = name_csv.replace('.png', '.csv')
    name_csv = name_csv.replace('.jpg', '.csv')
    name_csv = name_csv.replace('.jpeg', '.csv')

    with open(name_csv, 'w', newline='') as csvfile:
        field_names = ['Time', 'NumberObject', 'TypeObject', 'Positions']
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
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
