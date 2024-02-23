import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json
import time
import grpc
import torch
#from llava.serve.model_worker import main

def download_models():
    import os
    from huggingface_hub import snapshot_download


    model = os.getenv('MODEL', 'liuhaotian/llava-v1.6-mistral-7b')
    clip_model = 'openai/clip-vit-large-patch14-336'

    print(f'Downloading LLaVA model: {model}')
    snapshot_download(model)

    print(f'Downloading CLIP model: {model}')
    snapshot_download(clip_model)

if __name__ == "__main__":
    sys.execsys.exec