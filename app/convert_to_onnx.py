from ultralytics import YOLO


def main():
    model = YOLO("models/yolov8n.pt")
    model.export(
        format="onnx",
        imgsz=[480, 640],
        optimize=True,
        simplify=True,
    )


if __name__ == "__main__":
    main()
