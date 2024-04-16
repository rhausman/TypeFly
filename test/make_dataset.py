import json, sys, os
import time
import requests
import base64
from PIL import Image
sys.path.append("..")
sys.path.append("../controller")
from controller.yolo_client import YoloClient

from controller.llava_client import LlavaClient

print(os.getcwd())
BASE_URI = 'http://127.0.0.1:5000'
STREAM = False
# Get image locations
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, 'dataset')
RESULTS_PATH = os.path.join(DATASET_PATH, 'results.json')
ORIG_IMAGES_PATH = os.path.join(DATASET_PATH, 'original')
BOXES_IMAGES_PATH = os.path.join(DATASET_PATH, 'with_boxes')

yolo_client = YoloClient()

# Draw bounding boxes on the images, make a copy in with_boxes
if True: #len(os.listdir(BOXES_IMAGES_PATH)) < 2: # account for .DS_Store
    detect_results = {}
    image_names = os.listdir(ORIG_IMAGES_PATH)

    for name in image_names:
        image_path = os.path.join(ORIG_IMAGES_PATH, name)
        image = Image.open(image_path)
        yolo_client.detect_local(image)
        yolo_result = yolo_client.retrieve()
        if yolo_result is None:
            print(f"FAILED to retrieve result for {name}")
            continue
        
        # Plot bounding boxes, save image
        YoloClient.plot_results(yolo_result[0], yolo_result[1]['result'])
        yolo_result[0].save(os.path.join(BOXES_IMAGES_PATH, name))
        
        # save in the dictionary, and to the file. Ok to overwrite in each loop, speed is not a concern here.
        detect_results[name] = yolo_result[1]['result']
        with open(RESULTS_PATH, 'w') as f:
            json.dump(detect_results, f)
        print(f"Results saved for {name}: {yolo_result[1]['result']}.\nOther info: {yolo_result[1]}")
        


