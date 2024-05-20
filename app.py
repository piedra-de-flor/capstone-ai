from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

def detect_objects(image_files):
    model_path = 'C:/Users/USER/Desktop/공부/server_connect/best.onnx'
    model = YOLO(model_path)
    results_data = []

    person_class_id = None
    for key, value in model.names.items():
        if value == 'person':
            person_class_id = key
            break

    if (person_class_id is None):
        raise ValueError("Person class not found in the model's class names.")

    for filename in image_files:
        results = model(filename, project="server_connect", name="output")

        for res in results:
            person_count = sum(1 for box in res.boxes if box.cls == person_class_id)
            results_data.append({
                'detected_persons': person_count
            })

    return results_data

@app.route('/congestion', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join('C:/Users/USER/Desktop/공부/server_connect', file.filename)
        file.save(filename)
        results = detect_objects([filename])
        return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
