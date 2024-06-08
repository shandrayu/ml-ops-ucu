FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# To avoid questions :)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* 

RUN git clone https://github.com/zenseact/zod.git /app/third_party/zod \
    && cd /app/third_party/zod/zod \
    && pip install zod

RUN pip install ultralytics \
    && git clone https://github.com/THU-MIG/yolov10.git /app/third_party/yolo10 \
    && cd /app/third_party/yolo10 \
    && pip install . \
    && wget -P /app/data/yolo_weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt \
    && wget -P /app/data/yolo_weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt \
    && wget -P /app/data/yolo_weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt

RUN pip install --no-cache-dir grpcio grpcio-tools opencv-python-headless

COPY . /app

RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/object_detection_service.proto

ENV PYTHONPATH=/app

EXPOSE 50051

CMD ["python3", "grpc_server.py"]
