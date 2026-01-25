#!/usr/bin/env python
"""
Minimal ML API Server for 3D Print Failure Detection

This is a simplified version of obico-server's ml_api that only includes
the essential failure detection functionality.
"""

import flask
from flask_compress import Compress
from flask import abort, make_response, request, jsonify
from os import path, environ
import cv2
import numpy as np
import requests
import base64
from io import BytesIO

from detection_model import load_net, detect

# Detection threshold - boxes with confidence below this are ignored
THRESH = 0.08

app = flask.Flask(__name__)
Compress(app)

app.config['DEBUG'] = environ.get('DEBUG', 'False') == 'True'

# Load the model at startup
print("Loading failure detection model...")
model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(
    path.join(model_dir, 'model.cfg'),
    path.join(model_dir, 'model.meta')
)
print("Model loaded successfully!")


@app.route('/p/', methods=['GET', 'POST'])
def predict():
    """
    Analyze an image for 3D print failures.
    
    GET: Pass image URL as ?img=<url>
    POST: Pass image as base64 in JSON body {"image": "<base64>"}
          or as multipart form data with 'image' field
    
    Returns: {"detections": [{"label": "failure", "confidence": 0.85, "box": [x,y,w,h]}, ...]}
    """
    try:
        img = None
        
        if request.method == 'GET' and 'img' in request.args:
            # Fetch image from URL
            resp = requests.get(request.args['img'], stream=True, timeout=(5, 30))
            img_array = np.array(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            
        elif request.method == 'POST':
            if request.is_json:
                # Base64 encoded image in JSON
                data = request.get_json()
                if 'image' in data:
                    img_data = base64.b64decode(data['image'])
                    img_array = np.array(bytearray(img_data), dtype=np.uint8)
                    img = cv2.imdecode(img_array, -1)
            elif 'image' in request.files:
                # Multipart form upload
                file = request.files['image']
                img_array = np.array(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, -1)
        
        if img is None:
            return jsonify({
                'detections': [],
                'error': 'No valid image provided. Use ?img=<url> or POST with image data.'
            }), 400
        
        # Run detection
        detections = detect(net_main, img, thresh=THRESH)
        
        # Format response
        formatted_detections = []
        for det in detections:
            label, confidence, box = det
            formatted_detections.append({
                'label': label,
                'confidence': float(confidence),
                'box': [float(x) for x in box]
            })
        
        return jsonify({'detections': formatted_detections})
        
    except Exception as err:
        app.logger.error(f"Detection failed: {err}")
        return jsonify({
            'detections': [],
            'error': str(err)
        }), 500


@app.route('/hc/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': net_main is not None})


@app.route('/', methods=['GET'])
def index():
    """API info"""
    return jsonify({
        'name': '3D Print Failure Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/p/': 'POST image for failure detection',
            '/hc/': 'Health check'
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333, debug=True)
