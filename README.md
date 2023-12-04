# Trash Detection Camera
Experimental code for Seaside Robotics.

- Input
  - Camera image
- Output
  - Print trash detection result and send it via serial port

## How to develop

### Download model file
Put model file to `models/` directory.

https://github.com/tan-z-tan/trash_detection_camera/releases/download/onnx_model_2023-12-03/yolov8s.onnx

e.g.
```
wget https://github.com/tan-z-tan/trash_detection_camera/releases/download/onnx_model_2023-12-03/yolov8s.onnx -P models/
```

### Install dependencies
Install [poetry](https://python-poetry.org/docs/#installation) if you haven't installed.

Install python libraries.
```
poetry install
```

### Run
Test on your local computer (with debug visualization).
```
poetry run python -m app.main --visualize
```

If you specify the serial port, you can send the result to the microcomputer.
```
poetry run python -m app.main --serial-port /dev/serial0
```
