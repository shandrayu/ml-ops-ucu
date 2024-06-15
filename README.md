# ML Ops UCU Course Project. Visual Object Detection for Autonomous Driving

- [ML Ops UCU Course Project. Visual Object Detection for Autonomous Driving](#ml-ops-ucu-course-project-visual-object-detection-for-autonomous-driving)
  - [System design](#system-design)
  - [Video demonstration](#video-demonstration)
  - [Fine-tune YOLO 10](#fine-tune-yolo-10)
  - [Protobuf API](#protobuf-api)
  - [Docker](#docker)
  - [GRPC demo](#grpc-demo)

## System design

[System design doc link](system_design/ml_system_design_doc.md).

## Video demonstration

[Link to the video](https://drive.google.com/file/d/1SjBKqrDIgt-dsSnYpVi4ZGizcn7X4G7P/view?usp=sharing).

## Fine-tune YOLO 10

1. Install YOLO10 - see Docker file for details.

2. Train locally

   ```bash
   python scripts/run_training.py
   ```

## Protobuf API

Install dependencies

```bash
pip install grpcio grpcio-tools
```

Generate

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/object_detection_service.proto
```

## Docker

Nvidia docker

```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Build

```bash
docker build -t object-detection-service:v1.0 .  
```

Run bash

```bash
docker run -it --entrypoint /bin/bash object-detection-service:v1.0
```

Build && Run

> **Note:**
> All parameters, they are important for correct running of the container! All paths shall be full.

```bash
docker build -t object-detection-service:v1.0 . && docker run --gpus all --ipc=host -p 50051:50051 -v $(pwd)/data:/app/data -v $(pwd)/data:/$(pwd)/data -it object-detection-service:v1.0

```

## GRPC demo

```bash
python scripts/grpc_client.py
```
