from typing import Tuple

import numpy as np
import ultralytics
from pydantic import BaseModel

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
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float


class Detector:
    def __init__(
            self,
            model_path: str,
            classes: list[str],
            trash_classes: list[str],
            threshold: float = 0.5):
        self.model = ultralytics.YOLO(model_path)
        self.classes = classes
        self.trash_classes = trash_classes
        self.threshold = threshold
    
    def detect_objects(self, image: np.ndarray) -> list[Detection]:
        # 検出されたオブジェクトのリストを返す
        results: ultralytics.engine.results.Results = self.model.predict(source=image, stream=True)
        detections = []
        for res in results:
            detected_num = len(res.boxes.cls)
            for i in range(detected_num):
                label = self.classes[int(res.boxes.cls[i])]
                confidence = res.boxes.conf[i]
                if confidence < self.threshold:
                    continue
                if label not in self.trash_classes:
                    continue
                bbox = [int(v) for v in res.boxes.xyxy[i]]
                detections.append(Detection(label=label, confidence=confidence, bbox=bbox))
        return detections
