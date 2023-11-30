# Trash Detection Camera
Experimental code for Seaside Robotics.

- Input
  - Camera image
- Output
  - Print trash detection result and send it via serial port

## How to develop

### Download model file
Put model file to `models/` directory.
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

e.g.
```
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -P models/
```

### Install dependencies
```
poetry install
```

### Run
```
poetry run python -m app.main
```

If you specify the serial port, you can send the result to the microcomputer.
```
poetry run python -m app.main --serial-port /dev/ttyUSB0
```
