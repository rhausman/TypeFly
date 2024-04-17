import os, sys, json
import requests
from io import BytesIO

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
ROUTER_SERVICE_PORT = os.environ.get("ROUTER_SERVICE_PORT", "8888") #50049

"""
Access the Llava Service through http. Similar to YoLoClient.
"""
class LlavaClient: 
    llava_prompt_prefix = """You are the vision model for a drone that is attempting to accomplish tasks. 
    The planning module of the drone will use you to gather specific information about the visible scene by asking you a question. 
    When the question is asking for a specific piece of information, your answers should be as concise as possible, such as ‘which direction is person_20 facing?’ 
    (Answer should be ‘Right’ or ‘Left’.) However, when the question is open ended and less concrete, such as 'tell me about the scene in front of you', or ‘describe chair_1’, you can answer normally.
    Now, answer the following question from the drone: """
    def __init__(self, llava_prefix=False):
        print("init llava client")
        self.llava_prefix = llava_prefix
        self.service_url = f"http://{VISION_SERVICE_IP}:{ROUTER_SERVICE_PORT}/llava"
        # TODO: have llava client ping at startup to test if the service is available, 
        # and if not don't have it available as a skill?
    
    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def percieve_local(self, image, prompt):
        # send the image and prompt to the Llava service
        image_bytes = LlavaClient.image_to_bytes(image)
        print(f"percieve local processing. image: {image}, bytes: {image_bytes[:5]}")
        # Make a PromptRequest
        if self.llava_prefix:
            prompt = " ".join([LlavaClient.llava_prompt_prefix, prompt])
        files = {
            'image': ('image', image_bytes),
            'json_data': (None, json.dumps( {"prompt": prompt, 'service': 'llava', 'user_name': 'llava'}))
        }
        r = requests.post(self.service_url, files=files)
        return json.loads(r.text)
