import argparse
import json

import cv2
import serial

from app.detection.detector import (COCO_CLASSES, TRASH_CLASSES, Detection,
                                    Detector)


def capture_and_display(detector: Detector, serial_port: str | None = None, visualize: bool = False):
    cap = cv2.VideoCapture(0)  # カメラデバイスを開く

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO v8を使用してオブジェクトを検出
        objects = detector.detect_objects(frame)
        send_data(objects, serial_port=serial_port)

        if visualize:
            # 検出されたオブジェクトを画像上に描画
            for det in objects:
                cv2.rectangle(frame, (det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, det.label, (det.bbox[0], det.bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def send_data(detections: list[Detection], serial_port: str | None = '/dev/serial0', baud_rate: int = 9600):
    json_str = json.dumps([det.model_dump() for det in detections])
    print(json_str)

    if serial_port is not None:
        with serial.Serial(serial_port, baud_rate) as ser:
            for detection in detections:
                # シリアル通信でJSON形式の文字列としてデータを送信
                ser.write(json_str)


def main(serial_port: str | None = None, visualize: bool = False, threshold: float = 0.5):
    # Load a model
    detector = Detector(
        "models/yolov8m.pt",
        classes=COCO_CLASSES,
        trash_classes=TRASH_CLASSES,
        threshold=threshold)

    capture_and_display(
        detector=detector,
        serial_port=serial_port,
        visualize=visualize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial_port",
        type=str,
        help="Serial port name. If not specified, serial communication will not be used.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the detection result.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold of confidence score.",
        default=0.5,
    )
    args = parser.parse_args()

    main(args.serial_port, args.visualize, args.threshold)
