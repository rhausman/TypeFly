import json, sys, os
import time
import requests
import base64
from PIL import Image
sys.path.append("..")
sys.path.append("./controller")
print(os.getcwd())
from controller.llava_client import LlavaClient


BASE_URI = 'http://127.0.0.1:5000'
STREAM = False


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_elapsed_time(self):
        end = time.time()
        return round(end - self.start, 1)


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return str(base64.b64encode(image_file.read()).decode('utf-8'))


if __name__ == '__main__':
    print("Running test...")
    client = LlavaClient()
    
    image = Image.open('images/kitchen.webp')

    client.percieve_local(image, 'What is in the image?')
    #timer = Timer()