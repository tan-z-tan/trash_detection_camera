# Trash Detection Camera
Experimental code for Seaside Robotics.

## How to run

### Download model file
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

- Input
  - Camera image
- Output
  - Print trash detection result and send it via serial port

### Run
```
poetry run python -m app.main
```

If you specify the serial port, you can send the result to the microcomputer.
```
```
poetry run python -m app.main --serial-port /dev/ttyUSB0
```
