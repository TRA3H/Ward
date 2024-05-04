#!/usr/bin/env python3


from flask import Flask, request, jsonify
import os
import sys
import torch
sys.path.append('./')
from WardML import load_model, classify_image, load_class_index_mapping
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# List of allowed origins
origins = [
    "http://localhost:3000",
    "https://ward-web.netlify.app"
]

CORS(app, resources={r"/*": {"origins": origins}})

model = None  # Global variable to store the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_to_idx_path = './class_to_idx.json'
idx_to_class = load_class_index_mapping(class_to_idx_path)

@app.route('/')
def health_check():
    return jsonify({'status': 'up'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            if model is None:  # Load the model on first use
                model = load_model()
            result = classify_image(file.stream, model, device, idx_to_class)
            return jsonify({'predicted_class': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unsupported file type'}), 400

if __name__ == '__main__':
    # Ensuring the app runs on the correct port
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
