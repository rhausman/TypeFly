import os, sys, json
import requests
from io import BytesIO

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
ROUTER_SERVICE_PORT = os.environ.get("ROUTER_SERVICE_PORT", "8888") #50049

"""
Access the Llava Service through http. Similar to YoLoClient.
"""
class LlavaClient: 
    def __init__(self):
        self.service_url = f"http://{VISION_SERVICE_IP}:{ROUTER_SERVICE_PORT}/llava"
    
    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def percieve_local(self, image, prompt):
        # send the image and prompt to the Llava service
        image_bytes = LlavaClient.image_to_bytes(image)
        # Make a PromptRequest
        files = {
            'json_data': json.dumps( {"prompt": prompt}),
            'image': image_bytes
        }
        r = requests.post(self.service_url, data=files)
        return json.loads(r)
