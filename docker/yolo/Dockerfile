# Stage 1: Using Ubuntu for preparation
#FROM ubuntu:22.04 as builder
#FROM ultralytics/yolov5:m1-pyt-torch1.12-cpu
FROM ultralytics/yolov5:latest-arm64
# ultralytics/yolov5
RUN sed -i 's|ports.ubuntu.com/ubuntu-ports|old-releases.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update
RUN apt update
RUN apt install -y nano wget
RUN pip install grpcio-tools ultralytics

# copy the contents of the current project to /workspace
COPY ../.. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh

CMD ["python", "./serving/yolo/yolo_service.py"]