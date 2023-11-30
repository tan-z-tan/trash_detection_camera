import json
from typing import Tuple

import cv2
import serial
import ultralytics
from pydantic import BaseModel

# Load a model
model = ultralytics.YOLO("models/yolov8m.pt")


def capture_and_display(serial_port: str | None = None):
    cap = cv2.VideoCapture(0)  # カメラデバイスを開く

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO v8を使用してオブジェクトを検出
        objects = detect_objects(frame)
        if serial_port is not None:
            send_data_via_serial(objects, serial_port)

        # 検出されたオブジェクトを画像上に描画
        for det in objects:
            cv2.rectangle(frame, (det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, det.label, (det.bbox[0], det.bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


class Detection(BaseModel):
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float


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


def detect_objects(image) -> list[Detection]:
    # 検出されたオブジェクトのリストを返す
    results: ultralytics.engine.results.Results = model.predict(source=image, stream=True)  # don't save predictions
    detections = []
    for res in results:
        detected_num = len(res.boxes.cls)
        for i in range(detected_num):
            label = COCO_CLASSES[int(res.boxes.cls[i])]
            confidence = res.boxes.conf[i]
            if confidence < 0.5:
                continue
            if label not in TRASH_CLASSES:
                continue
            bbox = [int(v) for v in res.boxes.xyxy[i]]
            detections.append(Detection(label=label, confidence=confidence, bbox=bbox))

    return detections


def send_data_via_serial(detections: list[Detection], serial_port='/dev/ttyUSB0', baud_rate=9600):
    with serial.Serial(serial_port, baud_rate) as ser:
        for detection in detections:
            # シリアル通信でJSON形式の文字列としてデータを送信
            ser.write(json.dumps(detection.model_dump()).encode())


def main(serial_port: str | None = None):
    capture_and_display(serial_port=serial_port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial_port",
        type=str,
        help="Serial port name. If not specified, serial communication will not be used.",
        required=False,
    )
    args = parser.parse_args()
    main(args.serial_port)
