from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
import io
import face

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

    if person_class_id is None:
        raise ValueError("Person class not found in the model's class names.")

    for filename in image_files:
        results = model.predict(source=filename, save=True, project="C:/Users/USER/Desktop/공부/server_connect", name="output")

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

@app.route('/detect', methods=['POST'])
def upload_file_detect():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded files
    img1_path = os.path.join('uploads', file1.filename)
    img2_path = os.path.join('uploads', file2.filename)
    file1.save(img1_path)
    file2.save(img2_path)

    # Process the images using face.py
    try:
        result_image = face.find_most_similar_face(img1_path, img2_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Save the result image to a byte array
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/jpeg', as_attachment=True, download_name='result.jpg')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
