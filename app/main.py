import json

import cv2
import serial

from app.detection.detector import (COCO_CLASSES, TRASH_CLASSES, Detection,
                                    Detector)


def capture_and_display(detector: Detector, serial_port: str | None = None):
    cap = cv2.VideoCapture(0)  # カメラデバイスを開く

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO v8を使用してオブジェクトを検出
        objects = detector.detect_objects(frame)
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


def send_data_via_serial(detections: list[Detection], serial_port='/dev/serial0', baud_rate=9600):
    with serial.Serial(serial_port, baud_rate) as ser:
        for detection in detections:
            # シリアル通信でJSON形式の文字列としてデータを送信
            ser.write(json.dumps(detection.model_dump()).encode())


def main(serial_port: str | None = None):
    # Load a model
    detector = Detector(
        "models/yolov8m.pt",
        classes=COCO_CLASSES,
        trash_classes=TRASH_CLASSES,
        threshold=0.5)

    capture_and_display(
        detector=detector,
        serial_port=serial_port,
    )


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
