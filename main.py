from flask import Flask
from flask import request, jsonify
import cv2
import numpy as np
from util import detect, overlap


UPLOAD_FOLDER = './result'

app = Flask(__name__, static_folder='/', static_url_path='')


@app.route('/detect', methods=['POST'])
def detect_image():
    # read image
    file = request.files.get('image').read()
    image_numpy = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(image_numpy, 0)

    # pneumonia Detection
    bbox = detect(image)
    for box in bbox:
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 3)

    # save image
    cv2.imwrite('result/detect.jpg', image)

    return jsonify({'path' : 'result/detect.jpg'})

@app.route('/crop', methods=['POST'])
def crop_image():
    # read image
    file = request.files.get('image').read()
    image_numpy = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(image_numpy, 0)

    # get coordinates
    x1 = int(request.form['x1'])
    y1 = int(request.form['y1'])
    x2 = int(request.form['x2'])
    y2 = int(request.form['y2'])

    if x1 > x2:
        x1, x2 = x2, x1
    
    if y1 > y2:
        y1, y2 = y2, y1

    # pneumonia Detection
    bbox = detect(image)
    
    # check overlap
    result = False
    for box in bbox:
        if overlap(box, [x1, y1, x2, y2]):
            result = True

    return jsonify({'result' : result})

if __name__ == '__main__':
    # classifier = cv2.CascadeClassifier('model/cascade.xml')
    app.run('0.0.0.0', port= 8888, debug= True)