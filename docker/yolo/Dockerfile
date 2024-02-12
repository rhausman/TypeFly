FROM ultralytics/yolov5:latest-arm64
# ultralytics/yolov5
# Because this version of ubuntu (22.10) is no longer supported, we have to use an old package release
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
CMD ["python3", "./serving/yolo/yolo_service.py"]