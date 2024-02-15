FROM ultralytics/yolov5
# ultralytics/yolov5:latest-arm64
# Because this version of ubuntu (22.10) is no longer supported, we have to use an old package release
RUN sed -i 's|ports.ubuntu.com/ubuntu-ports|old-releases.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update
RUN apt update
RUN apt install -y nano wget
RUN pip install grpcio-tools ultralytics
#RUN apt-get install g++-aarch64-linux-gnu -y
RUN apt-get install g++ -y
RUN pip3 install lap

# copy the contents of the current project to /workspace
COPY ../.. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh
#RUN python3 -m http.server 50050 --bind 0.0.0.0
EXPOSE 8888
#CMD ["python3", "./serving/yolo/yolo_service.py"]
CMD ["python3", "-m", "http.server", "8888", "--bind", "0.0.0.0"]