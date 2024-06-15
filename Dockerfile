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
    && rm -rf /var/lib/apt/lists/* 

# TODO: replace to specific repo revision
RUN git clone https://github.com/zenseact/zod.git /app/third_party/zod \
    && cd /app/third_party/zod/zod \
    && pip3 install zod

# TODO: replace to specific repo revision
RUN git clone https://github.com/THU-MIG/yolov10.git /app/third_party/yolo10

# setup.py for some reason is missing from yolo10 repo. 
# We need it in order to correctly install modified ulralytics.
COPY ./ultralytics/setup.py /app/third_party/yolo10

RUN cd /app/third_party/yolo10 \
    && pip3 install -r requirements.txt \
    && pip3 install -e .

# TODO: not used now, the functionality was not checked
# Dockerfile debugging
RUN pip3 install ptvsd

COPY requirements.txt /app/

# TODO: replace to specific package versions, now packages in requirements.txt are without versions
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/object_detection_service.proto

ENV PYTHONPATH=/app

ENV TORCH_HOME=/app/torch_home

# Expose port 50051 for the gRPC service and 5678 for the debugger
EXPOSE 50051 5678

CMD ["python3", "grpc_server.py"]
