import argparse
import json

import cv2
import serial

from app.detection.detector_onnx import (COCO_CLASSES, TRASH_CLASSES, Detection, DetectorOnnx)


def capture_and_display(detector: DetectorOnnx, serial_port: str | None = None, visualize: bool = False):
    cap = cv2.VideoCapture(0)  # カメラデバイスを開く
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # カメラ画像の横幅を640に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # カメラ画像の縦幅を480に設定

    while True:
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            break

        # YOLO v8を使用してオブジェクトを検出
        detections = detector.detect_objects(frame)
        send_data(detections, serial_port=serial_port)

        if visualize:
            # 検出されたオブジェクトを画像上に描画
            for det in detections:
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
                ser.write(json_str.encode('utf-8'))


def main(serial_port: str | None = None, visualize: bool = False, threshold: float = 0.5):
    # Load a model
    detector = DetectorOnnx(
        # "models/yolov8m.onnx",
        "models/yolov8s.onnx",
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
