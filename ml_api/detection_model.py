#!/usr/bin/env python
"""
Simplified detection model loader - ONNX only for maximum compatibility.
Based on obico-server's ml_api/lib/detection_model.py
"""

import cv2
import numpy as np
from os import path

# Global for class names
alt_names = None


class OnnxNet:
    """ONNX Runtime based neural network for failure detection"""
    
    def __init__(self, weights_path: str, meta_path: str, use_gpu: bool = False):
        import onnxruntime as ort
        
        # Configure execution providers
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Load metadata
        self.meta = Meta(meta_path)
        
        # Get expected input dimensions
        self.net_width = self.input_shape[3] if len(self.input_shape) == 4 else 416
        self.net_height = self.input_shape[2] if len(self.input_shape) == 4 else 416
        
        print(f"ONNX model loaded: input shape {self.input_shape}")
    
    def force_cpu(self):
        """Force CPU execution"""
        import onnxruntime as ort
        self.session.set_providers(['CPUExecutionProvider'])
    
    def detect(self, meta, image, names, thresh=0.5, hier_thresh=0.5, nms=0.45, debug=False):
        """Run detection on an image"""
        # Preprocess image
        h, w = image.shape[:2]
        
        # Resize and normalize
        resized = cv2.resize(image, (self.net_width, self.net_height))
        
        # Convert BGR to RGB and normalize to 0-1
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # NCHW format (batch, channels, height, width)
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Parse outputs - format depends on model export
        detections = self._parse_outputs(outputs, w, h, thresh, nms, names)
        
        return detections
    
    def _parse_outputs(self, outputs, orig_w, orig_h, thresh, nms_thresh, names):
        """Parse ONNX model outputs into detection list"""
        detections = []
        
        # The output format varies by model - this handles common YOLO formats
        output = outputs[0]
        
        if len(output.shape) == 3:
            # Shape: [1, num_detections, 5+num_classes] or [1, 5+num_classes, num_detections]
            output = output[0]
            
            # Check if we need to transpose
            if output.shape[0] < output.shape[1]:
                output = output.T
            
            boxes = []
            confidences = []
            class_ids = []
            
            for detection in output:
                if len(detection) >= 5:
                    # YOLO format: [x, y, w, h, obj_conf, class1_conf, class2_conf, ...]
                    x, y, w, h = detection[:4]
                    obj_conf = detection[4] if len(detection) > 4 else 1.0
                    
                    if len(detection) > 5:
                        class_scores = detection[5:]
                        class_id = np.argmax(class_scores)
                        confidence = obj_conf * class_scores[class_id]
                    else:
                        class_id = 0
                        confidence = obj_conf
                    
                    if confidence >= thresh:
                        # Convert from relative to absolute coordinates
                        abs_x = x * orig_w
                        abs_y = y * orig_h
                        abs_w = w * orig_w
                        abs_h = h * orig_h
                        
                        # Convert from center to corner format for NMS
                        left = abs_x - abs_w / 2
                        top = abs_y - abs_h / 2
                        
                        boxes.append([left, top, abs_w, abs_h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply NMS
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms_thresh)
                
                for i in indices:
                    idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                    box = boxes[idx]
                    label = names[class_ids[idx]] if class_ids[idx] < len(names) else 'failure'
                    detections.append((label, confidences[idx], tuple(box)))
        
        return detections


class Meta:
    """Metadata loader for model configuration"""
    
    def __init__(self, meta_path: str):
        self.names = []
        self.num_classes = 0
        
        try:
            with open(meta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('names'):
                        # Parse: names = path/to/names.txt
                        names_file = line.split('=')[1].strip()
                        names_dir = path.dirname(meta_path)
                        names_path = path.join(names_dir, names_file)
                        
                        if path.exists(names_path):
                            with open(names_path, 'r') as nf:
                                self.names = [n.strip() for n in nf.readlines() if n.strip()]
                    elif line.startswith('classes'):
                        self.num_classes = int(line.split('=')[1].strip())
        except Exception as e:
            print(f"Warning: Could not load meta file: {e}")
            self.names = ['failure']  # Default class name
            self.num_classes = 1


def load_net(config_path: str, meta_path: str, weights_path: str | None = None):
    """
    Load the detection model.
    
    Args:
        config_path: Path to model config (not used for ONNX but kept for compatibility)
        meta_path: Path to metadata file with class names
        weights_path: Optional explicit path to weights file
    
    Returns:
        Loaded network ready for inference
    """
    global alt_names
    
    # Prioritized list of weight locations to try
    weight_locations = [
        '/model_cache/ml_api/onnx/model-weights.onnx',
        path.join(path.dirname(meta_path), 'model-weights.onnx'),
    ]
    
    if weights_path:
        weight_locations.insert(0, weights_path)
    
    # Try to load from each location
    for weights in weight_locations:
        if path.exists(weights):
            try:
                print(f"Loading ONNX model from: {weights}")
                net = OnnxNet(weights, meta_path, use_gpu=False)
                
                # Set global names
                alt_names = net.meta.names
                
                return net
            except Exception as e:
                print(f"Failed to load {weights}: {e}")
                continue
    
    raise Exception(f"Could not load model from any location: {weight_locations}")


def detect(net, image, thresh=0.5, hier_thresh=0.5, nms=0.45, debug=False):
    """Run detection on an image using the loaded network"""
    return net.detect(net.meta, image, alt_names or ['failure'], thresh, hier_thresh, nms, debug)
