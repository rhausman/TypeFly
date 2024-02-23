import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json
import time
import grpc
import torch
import multiprocessing

sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "LLaVA")))

from llava.model.builder import load_pretrained_model
import multiprocessing

MODEL_PATH = os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.6-mistral-7b") # /liuhautian/...
MODEL_NAME = os.environ.get("MODEL_NAME", "llava-v1.6-mistral-7b")
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
CWD = os.getcwd()
SERVICE_PORT = os.environ.get("LLAVA_SERVICE_PORT", "50058, 50059").split(",")

sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

"""
gRPC service class.
For now, LlavaService will not have a Stream mode. It simply processes a frame and prompt and 
returns the result.
"""

def load_model():
    
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    """
    device = torch.device('cpu') # todo delete. cuda:5
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=MODEL_PATH, model_base=None, model_name=MODEL_NAME, load_8bit=False, load_4bit=False, device=device)
    model.to(device)
    print(f"GPU memory usage: {torch.cuda.memory_allocated()}")
    return tokenizer, model, image_processor, context_len

def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

class LlavaService(hyrch_serving_pb2_grpc.LlavaServiceServicer):
    def __init__(self, port):
        self.port = port
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model()
    
    def reload_model(self):
        release_model(self.model)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model()
    
    # PromptRequest must contain image data for LLaVA
    @staticmethod
    def bytes_to_image(image_bytes):
        return Image.open(BytesIO(image_bytes))
    
    """
    Recieve a Percieve Request. request is a promptrequest, which contains a string and an image.
    Returns: PromptResponse.
    """
    def Percieve(self, request, context):
        print(f"Received Percieve request from {context.peer()} on port {self.port}, prompt: {request.json_data}")
        prompt = request.json_data # maybe should rename this field to prompt or use the json with json.loads
        if prompt is None or prompt == "":
            print("No prompt provided. Returning empty response.")
            return hyrch_serving_pb2.PromptResponse(json_data="No prompt provided. Continue completing the task.")
        image = self.bytes_to_image(request.image)
        #temperature = request.temperature
        #max_new_tokens = request.max_new_tokens
        # todo: use the model to make a prediction
        raise NotImplementedError("Not implemented!")
        return hyrch_serving_pb2.PromptResponse(json_data="Not implemented!")
      
# TODO remove  
tokenizer, model, image_processor, _ = load_model()
release_model(model)