import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json
import time
import grpc
import torch
from ultralytics import YOLO
import multiprocessing

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
SERVICE_PORT = os.environ.get("LLAVA_SERVICE_PORT", "50058, 50059").split(",")


