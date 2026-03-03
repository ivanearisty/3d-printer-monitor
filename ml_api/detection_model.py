"""Simplified detection model loader - ONNX only for maximum compatibility.

Supports two model formats:
- YOLOv4 (Obico/Darknet): 2 output tensors [boxes, confidences]
- YOLOv8/v11 (Ultralytics): 1 output tensor [1, 4+num_classes, num_detections]

The model format is auto-detected based on the number of ONNX output tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import cv2  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Global for class names
alt_names: list[str] | None = None

Detection: TypeAlias = tuple[str, float, tuple[float, float, float, float]]


def nms_cpu(
    boxes: NDArray[np.floating[Any]],
    confs: NDArray[np.floating[Any]],
    nms_thresh: float = 0.5,
    *,
    min_mode: bool = False,
) -> NDArray[np.intp]:
    """Perform non-maximum suppression on CPU."""
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


def post_processing_yolov4(
    output: list[NDArray[np.floating[Any]]],
    width: int,
    height: int,
    conf_thresh: float,
    nms_thresh: float,
    names: list[str],
) -> list[list[Detection]]:
    """Post-process YOLOv4 ONNX outputs (2 tensors: boxes + confidences)."""
    box_array = output[0]
    confs = output[1]

    if type(box_array).__name__ != "ndarray":
        box_array = box_array.cpu().detach().numpy()  # type: ignore[attr-defined]
        confs = confs.cpu().detach().numpy()  # type: ignore[attr-defined]

    num_classes = confs.shape[2]

    # Select first coordinate set from box array
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    def box_x1x1x2y2_to_xcycwh_scaled(
        b: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        return (
            float(0.5 * width * (b[0] + b[2])),
            float(0.5 * height * (b[1] + b[3])),
            float(width * (b[2] - b[0])),
            float(height * (b[3] - b[1])),
        )

    dets_batch: list[list[Detection]] = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes: list[list[float]] = []
        # nms for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                bboxes.extend(
                    [
                        ll_box_array[k, 0], ll_box_array[k, 1],
                        ll_box_array[k, 2], ll_box_array[k, 3],
                        ll_max_conf[k], ll_max_conf[k], ll_max_id[k],
                    ]
                    for k in range(ll_box_array.shape[0])
                )

        detections: list[Detection] = [
            (names[int(b[6])], float(b[4]), box_x1x1x2y2_to_xcycwh_scaled((b[0], b[1], b[2], b[3])))
            for b in bboxes
        ]
        dets_batch.append(detections)

    return dets_batch


def post_processing_yolo11(
    output: list[NDArray[np.floating[Any]]],
    input_w: int,
    input_h: int,
    width: int,
    height: int,
    conf_thresh: float,
    nms_thresh: float,
    names: list[str],
) -> list[list[Detection]]:
    """Post-process YOLOv8/v11 ONNX output (single tensor).

    Output shape: [1, 4 + num_classes, num_detections]
    - First 4 rows: x_center, y_center, w, h (in pixels relative to input size)
    - Remaining rows: class confidence scores
    """
    raw = output[0]  # shape: [1, 4+C, N]
    predictions = raw[0].T  # shape: [N, 4+C]

    num_classes = predictions.shape[1] - 4
    boxes_xywh = predictions[:, :4]  # x_center, y_center, w, h in input-size pixels
    class_scores = predictions[:, 4:]

    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Filter by confidence
    mask = max_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(max_scores) == 0:
        return [[]]

    # Convert xywh (input-pixel coords) to x1y1x2y2 for NMS
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Scale factors from input size to original image size
    sx = width / input_w
    sy = height / input_h

    # NMS per class
    all_detections: list[Detection] = []
    for j in range(num_classes):
        cls_mask = class_ids == j
        if not np.any(cls_mask):
            continue

        cls_boxes = xyxy[cls_mask]
        cls_scores_j = max_scores[cls_mask]
        cls_xywh = boxes_xywh[cls_mask]

        keep = nms_cpu(cls_boxes, cls_scores_j, nms_thresh)
        if keep.size == 0:
            continue

        for k in keep:
            # Scale xywh to original image coordinates
            xc = float(cls_xywh[k, 0] * sx)
            yc = float(cls_xywh[k, 1] * sy)
            w = float(cls_xywh[k, 2] * sx)
            h = float(cls_xywh[k, 3] * sy)
            label = names[j] if j < len(names) else f"class_{j}"
            all_detections.append((label, float(cls_scores_j[k]), (xc, yc, w, h)))

    return [all_detections]


class OnnxNet:
    """ONNX Runtime based neural network for failure detection."""

    def __init__(self, weights_path: str, meta_path: str, use_gpu: bool = False) -> None:
        """Initialize the ONNX model from weights and metadata files."""
        import onnxruntime as ort

        # Configure execution providers
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.meta = Meta(meta_path)

        # Auto-detect model format based on number of output tensors
        num_outputs = len(self.session.get_outputs())
        self.model_format = "yolo11" if num_outputs == 1 else "yolov4"

        print(f"ONNX model loaded ({self.model_format} format)")
        print(f"  {len(self.session.get_inputs())} inputs, {num_outputs} outputs")
        for inp in self.session.get_inputs():
            print(f"  Input: {inp.name} shape={inp.shape}")
        for out in self.session.get_outputs():
            print(f"  Output: {out.name} shape={out.shape}")

    def force_cpu(self) -> None:
        """Force CPU execution."""
        self.session.set_providers(["CPUExecutionProvider"])

    def detect(
        self,
        meta: Meta,
        image: NDArray[np.uint8],
        alt_names_arg: list[str] | None,
        thresh: float = 0.5,
        _hier_thresh: float = 0.5,
        nms: float = 0.45,
        *,
        debug: bool = False,
    ) -> list[Detection]:
        """Run detection on an image.

        Auto-dispatches to YOLOv4 or YOLOv8/v11 post-processing.
        """
        input_h = self.session.get_inputs()[0].shape[2]
        input_w = self.session.get_inputs()[0].shape[3]
        width = image.shape[1]
        height = image.shape[0]

        # Input preprocessing
        resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0  # type: ignore[misc]

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_in})

        if debug:
            print(f"Model outputs: {len(outputs)} tensors")
            for i, out in enumerate(outputs):
                print(f"  Output {i}: shape={out.shape}, min={out.min():.4f}, max={out.max():.4f}")  # type: ignore[union-attr]

        names = meta.names or alt_names_arg or ["failure"]

        if self.model_format == "yolo11":
            detections = post_processing_yolo11(outputs, input_w, input_h, width, height, thresh, nms, names)
        else:
            detections = post_processing_yolov4(outputs, width, height, thresh, nms, names)

        return detections[0]


class Meta:
    """Metadata loader for model configuration."""

    def __init__(self, meta_path: str) -> None:
        """Load model metadata from the given path."""
        self.names: list[str] = []
        self.num_classes: int = 0

        try:
            with Path(meta_path).open() as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("names"):
                        names_file = stripped.split("=")[1].strip()
                        names_dir = Path(meta_path).parent
                        names_path = names_dir / names_file

                        if names_path.exists():
                            with names_path.open() as nf:
                                self.names = [n.strip() for n in nf if n.strip()]
                    elif stripped.startswith("classes"):
                        self.num_classes = int(stripped.split("=")[1].strip())
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load meta file: {e}")
            self.names = ["failure"]  # Default class name
            self.num_classes = 1

        if not self.names:
            self.names = ["failure"]
            self.num_classes = 1


def load_net(_config_path: str, meta_path: str, weights_path: str | None = None) -> OnnxNet:
    """Load the detection model.

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
        "/model_cache/ml_api/onnx/model-weights.onnx",
        str(Path(meta_path).parent / "model-weights.onnx"),
    ]

    if weights_path:
        weight_locations.insert(0, weights_path)

    # Try to load from each location
    for weights in weight_locations:
        if Path(weights).exists():
            try:
                print(f"Loading ONNX model from: {weights}")
                net = OnnxNet(weights, meta_path, use_gpu=False)

                # Set global names
                alt_names = net.meta.names

                return net
            except Exception as e:  # noqa: BLE001
                print(f"Failed to load {weights}: {e}")
                continue

    raise FileNotFoundError(f"Could not load model from any location: {weight_locations}")


def detect(
    net: OnnxNet,
    image: NDArray[np.uint8],
    thresh: float = 0.5,
    hier_thresh: float = 0.5,
    nms: float = 0.45,
    *,
    debug: bool = False,
) -> list[Detection]:
    """Run detection on an image using the loaded network."""
    return net.detect(net.meta, image, alt_names or ["failure"], thresh, hier_thresh, nms, debug=debug)
