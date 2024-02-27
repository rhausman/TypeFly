import sys, os, json
from quart import Quart, request, jsonify
import asyncio

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
ROUTER_SERVICE_PORT = os.environ.get("ROUTER_SERVICE_PORT", "50049")

sys.path.append(os.path.join(ROOT_PATH, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

from service_manager import ServiceManager

app = Quart(__name__)

grpcServiceManager = ServiceManager()

service_lock = asyncio.Lock()

@app.before_serving
async def before_serving():
    global grpcServiceManager
    grpcServiceManager.add_service("yolo", os.environ.get("VISION_SERVICE_IPS", "localhost"), os.environ.get("YOLO_SERVICE_PORT", "50050, 50051"))
    grpcServiceManager.add_service("llava", os.environ.get("VISION_SERVICE_IPS", "localhost"), os.environ.get("LLAVA_SERVICE_PORT", "5000"))
    await grpcServiceManager._initialize_channels()

@app.route('/yolo', methods=['POST'])
async def process_yolo():
    print("Got yolo request")
    global grpcServiceManager
    global service_lock
    files = await request.files
    form = await request.form
    image_data = files.get('image')
    json_str = form.get('json_data')

    print(f"Received request with json_data: {json_str} and files: {files}")

    if not json_str:
        return "No JSON data provided", 400
    
    if not image_data:
        return "No image provided", 400
    
    json_data = json.loads(json_str)
    user_name = json_data.get("user_name", "user")
    stream_mode = json_data.get("stream_mode", False)
    image_id = json_data.get("image_id", None)

    async with service_lock:
        channel = await grpcServiceManager.get_service_channel("yolo", dedicated=stream_mode, user_name=user_name)

    try:
        stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        image_contents = image_data.read()
        if stream_mode:
            response = await stub.DetectStream(hyrch_serving_pb2.DetectRequest(image_id=image_id, image_data=image_contents))
        else:
            response = await stub.Detect(hyrch_serving_pb2.DetectRequest(image_id=image_id, image_data=image_contents))
    finally:
        if not stream_mode:
            await grpcServiceManager.release_service_channel("yolo", channel)
    return response.json_data

@app.route('/llava', methods=['POST'])
async def process_llava():
    print("Got llava request")
    global grpcServiceManager
    global service_lock
    files = await request.files
    form = await request.form
    image_data = files.get('image')
    json_str = form.get('json_data')

    print(f"Received request with json_data: {json_str}, and img data: {image_data}")

    if not json_str:
        return "No JSON data provided", 400
    
    if not image_data:
        return "No image provided", 400
    
    json_data = json.loads(json_str)
    prompt = json_data.get("prompt", None)
    print("Getting service lock")
    async with service_lock:
        channel = await grpcServiceManager.get_service_channel("llava", dedicated=False, user_name="user")
    print("Got service lock, attempting request")
    try:
        stub = hyrch_serving_pb2_grpc.LlavaServiceStub(channel)
        image_contents = image_data.read()
        print("Sending request to llava")
        response = stub.Percieve(hyrch_serving_pb2.PromptRequest(prompt=prompt, image_data=image_contents))
    finally:
        await grpcServiceManager.release_service_channel("llava", channel)
    print("Done response from llava")
    return response.json_data

@app.route('/testing', methods=['GET'])
async def process_test():
    print('processing test request')
    return "success"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=ROUTER_SERVICE_PORT)