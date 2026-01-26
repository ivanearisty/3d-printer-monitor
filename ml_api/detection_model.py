#!/usr/bin/env python
"""
Simplified detection model loader - ONNX only for maximum compatibility.
Based on obico-server's ml_api/lib/detection_model.py and ml_api/lib/onnx.py
"""

import cv2  # type: ignore[import-untyped]
import numpy as np
from os import path
from typing import List, Tuple

# Global for class names
alt_names = None


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    """Non-maximum suppression on CPU"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(output, width, height, conf_thresh, nms_thresh, names):
    """Post-process ONNX model outputs - matches obico-server's format"""
    box_array = output[0]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    def box_x1x1x2y2_to_xcycwh_scaled(b):
        return (
            float(0.5 * width * (b[0] + b[2])),
            float(0.5 * height * (b[1] + b[3])),
            float(width * (b[2] - b[0])),
            float(height * (b[3] - b[1]))
        )

    dets_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        detections = [(names[int(b[6])], float(b[4]), box_x1x1x2y2_to_xcycwh_scaled((b[0], b[1], b[2], b[3]))) for b in bboxes]
        dets_batch.append(detections)

    return dets_batch


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
        self.meta = Meta(meta_path)
        
        print(f"ONNX model loaded with {len(self.session.get_inputs())} inputs, {len(self.session.get_outputs())} outputs")
        for inp in self.session.get_inputs():
            print(f"  Input: {inp.name} shape={inp.shape}")
        for out in self.session.get_outputs():
            print(f"  Output: {out.name} shape={out.shape}")
    
    def force_cpu(self):
        """Force CPU execution"""
        import onnxruntime as ort
        self.session.set_providers(['CPUExecutionProvider'])
    
    def detect(self, meta, image, alt_names, thresh=0.5, hier_thresh=0.5, nms=0.45, debug=False) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """Run detection on an image - matches obico-server's OnnxNet.detect"""
        input_h = self.session.get_inputs()[0].shape[2]
        input_w = self.session.get_inputs()[0].shape[3]
        width = image.shape[1]
        height = image.shape[0]

        # Input preprocessing
        resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_in})
        
        if debug:
            print(f"Model outputs: {len(outputs)} tensors")
            for i, out in enumerate(outputs):
                print(f"  Output {i}: shape={out.shape}, min={out.min():.4f}, max={out.max():.4f}")  # type: ignore[union-attr]

        detections = post_processing(outputs, width, height, thresh, nms, meta.names)
        return detections[0]


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
        
        if not self.names:
            self.names = ['failure']
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
