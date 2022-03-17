import cv2
import numpy as np

def iou(first_box, second_box):
    """"
        Format: [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    """
    x_left = max(first_box[0], second_box[0])
    y_top = max(first_box[1], second_box[1])
    x_right = min(first_box[2], second_box[2])
    y_bottom = min(first_box[3], second_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)

    first_box_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_box_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])

    union = first_box_area + second_box_area - intersection

    return intersection / union

def overlap(first_box, second_box):
    """"
        Format: [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    """
    x_left = max(first_box[0], second_box[0])
    y_top = max(first_box[1], second_box[1])
    x_right = min(first_box[2], second_box[2])
    y_bottom = min(first_box[3], second_box[3])

    if x_right < x_left or y_bottom < y_top:
        intersection = 0
    else:
        intersection = (x_right - x_left) * (y_bottom - y_top)
    
    first_box_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_box_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])

    if (intersection / first_box_area >= 0.5) or (intersection / second_box_area >= 0.5):
        return True
    
    return False

def remove_overlap(boxes):
    """"
        boxes include numpy arrays
        Format: [x, y, w, h]
    """
    overlapping = True
    while(overlapping):
        overlapping = False
        boxes_tmp = []
        for i in range(len(boxes)):
            if overlapping:
                break
            for j in range(i + 1, len(boxes)):
                first_box = [boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]]
                second_box = [boxes[j][0], boxes[j][1], boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]]
                if overlap(first_box, second_box):
                    overlapping = True
                    for k in range(len(boxes)):
                        if k != i and k != j:
                            boxes_tmp.append(boxes[k])
                    x_left = min(first_box[0], second_box[0])
                    y_top = min(first_box[1], second_box[1])
                    x_right = max(first_box[2], second_box[2])
                    y_bottom = max(first_box[3], second_box[3])
                    boxes_tmp.append(np.array([x_left, y_top, x_right - x_left, y_bottom - y_top]))
                    boxes = boxes_tmp
                    break
        
    return np.array(boxes)

def conver_format(boxes):
    result = []
    for box in boxes:
        result.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    
    return np.array(result)

def detect(image):
    classifier = cv2.CascadeClassifier('model/cascade.xml')
    bbox = classifier.detectMultiScale(image)
    bbox = remove_overlap(bbox)
    bbox = conver_format(bbox)
    
    return bbox

def crop(image, x, y, width, height):
    pass

if __name__ == '__main__':
    a = np.array([0, 0, 100, 100])
    b = np.array([25, 25, 150, 150])
    c = np.array([250, 250, 50, 50])
    bbox = remove_overlap(np.array([a, b, c]))
    for box in bbox:
        print(box)