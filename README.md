# ML Ops UCU Course Project. Visual Object Detection for Autonomous Driving

- [ML Ops UCU Course Project. Visual Object Detection for Autonomous Driving](#ml-ops-ucu-course-project-visual-object-detection-for-autonomous-driving)
  - [System design](#system-design)
  - [Video demonstration](#video-demonstration)
  - [Fine-tune YOLO 10](#fine-tune-yolo-10)
  - [Protobuf API](#protobuf-api)
  - [Docker](#docker)
  - [GRPC demo](#grpc-demo)
  - [Experiment tracking](#experiment-tracking)
    - [Summary](#summary)
    - [Setup](#setup)
      - [Minio](#minio)
      - [Generate docker compose environment file](#generate-docker-compose-environment-file)
      - [Docker compose](#docker-compose)

## System design

[System design doc link](system_design/ml_system_design_doc.md).

## Video demonstration

[Containerization video](https://drive.google.com/file/d/1-1YFQxsPfVtP2vcepKyU_m1efNmzZ8Z9/view?usp=sharing).
[MLFlow+Minio in docker video](https://drive.google.com/file/d/1H321j42NHZjzOTuLzbBEXu28W7vtUCUi/view?usp=sharing).

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

## Experiment tracking

### Summary

If secrets are already generated, build image and run with command

```bash
docker build -t object-detection-service:v1.0 . && docker-compose up -d
```

```bash
docker compose down
```

MLFlow address http://127.0.0.1:5000.

### Setup

#### Minio

- Run minio
- Login to the console
- Generate credentials
- Save credentials to `credentials.json`.

```bash
docker pull minio/minio
```

```bash
docker volume create minio-data
```

To run independently (usually not needed because it is done in docker-compose):

```bash
docker run -d --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v minio-data:/data \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

You can now access the Minio server using the browser at (http://localhost:9000)[http://localhost:9000] and the Minio console at [http://localhost:9001](http://localhost:9001) using the credentials `minioadmin` for both the access key and secret key.

#### Generate docker compose environment file

```bash
./generate_env.sh
```

#### Docker compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
docker-compose --version
```
