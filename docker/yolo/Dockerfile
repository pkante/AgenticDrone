FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        nano \
        wget \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
        ultralytics \
        grpcio \
        grpcio-tools \
        lapx \
        opencv-python \
        Pillow \
        numpy

# copy the contents of the current project to /workspace
COPY ../.. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh

CMD ["python", "./serving/yolo/yolo_service.py"]
