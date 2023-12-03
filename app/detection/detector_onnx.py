from typing import Tuple

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel

from app.detection.utils import multiclass_nms, xywh2xyxy

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
TRASH_CLASSES = [
    "bottle",
    "wine glass",
    "vase",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "book",
    "toothbrush",
]


class Detection(BaseModel):
    label: str  # class name
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float  # 0.0 ~ 1.0


class DetectorOnnx:
    def __init__(
            self,
            model_path: str,
            classes: list[str],
            trash_classes: list[str],
            threshold: float = 0.5):
        self.model = ort.InferenceSession(model_path)
        self.classes = classes
        self.trash_classes = trash_classes
        self.threshold = threshold
        self.iou_threshold = 0.5

    def detect_objects(self, image: np.ndarray) -> list[Detection]:
        # 検出されたオブジェクトのリストを返す
        # preprocess
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        image /= 255.0
        image = np.ascontiguousarray(image)

        # inference
        outputs = self.model.run(["output0"], {"images": image})
        detections = self.process_output(outputs)

        return detections

    def process_output(self, outputs) -> list[Detection]:
        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.threshold, :]
        scores = scores[scores > self.threshold]

        if len(scores) == 0:
            return []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        detections: list[Detection] = []
        for idx in indices:
            detections.append(
                Detection(
                    label=self.classes[class_ids[idx]],
                    bbox=[int(v) for v in boxes[idx]],
                    confidence=scores[idx])
                )
        return detections

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        # boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes
