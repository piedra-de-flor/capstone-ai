from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

def detect_objects(image_files):
    model_path = 'C:/Users/USER/Desktop/공부/server_connect/best.onnx'
    model = YOLO(model_path)
    results_list = []

    for filename in image_files:
        results = model(filename, project="server_connect", name="output")

        for i, res in enumerate(results):
            result_image_path = os.path.join('C:/Users/USER/Desktop/공부/server_connect', f'result_{i}.jpg')
            res.save(result_image_path)

        image_boxes = []
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            image_boxes.extend(boxes)
        results_list.append(image_boxes)

    return results_list

@app.route('/congestion', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        results = detect_objects([file.filename])
        return jsonify({'result': results})

if __name__ == '__main__':
    app.run(debug=True)
