import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json, time, grpc, multiprocessing
import torch
from transformers.generation.streamers import TextStreamer, TextIteratorStreamer

sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "LLaVA")))

from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

MODEL_PATH = os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.6-mistral-7b") # /liuhautian/...
MODEL_NAME = os.environ.get("MODEL_NAME", "llava-v1.6-mistral-7b")
MODEL_BASE = os.environ.get("MODEL_BASE", None)
CONV_MODE = os.environ.get("CONV_MODE", "mistral_instruct")

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
CWD = os.getcwd()
SERVICE_PORT = os.environ.get("LLAVA_SERVICE_PORT", "5000") # .split(",")

sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
"""
gRPC service class.
For now, LlavaService will not have a Stream mode. It simply processes a frame and prompt and 
returns the result.
"""

def load_model():
    print(f"Cuda status... available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    
    print(f"loading_model on device: {device}")
    #blockPrint()
    print("test: shouldn't print here")
    device = torch.device(device) # todo delete. cuda:5
    # todo: cache the download
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=MODEL_PATH, model_base=MODEL_BASE, model_name=MODEL_NAME, load_8bit=False, load_4bit=False, device=device)
    model.to(device)
    #enablePrint()
    print("Test: should print")
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
    
    @staticmethod
    def format_outputs(outputs):
        # Replace special tokens etc and join the outputs
        print(f"formatting outputs... {type(outputs)}")
        out_list = []
        for output in outputs:
            print(f"output: {type(output)}, {output.__dict__}")
            print(f"content: {output}")
            out_list.append(output.replace('<s>', '').replace('</s>', '').strip())
        #s = "\n".join([output.replace('<s>', '').replace('</s>', '').strip() for output in outputs])
        s = "\n".join(out_list)
        return s
        
    # based on run_inference in LLaVA/llava/serve/api.py, simplified for our use case.
    def run_inference(self, data):
        """
        Run inference on the model.
        """
        # Todo infer the conv mode from model name rather than global constant
        # Get a copy of the conversation template
        conv = conv_templates[CONV_MODE].copy()
        # Process the image, load into tensor
        prompt = data['prompt']
        image = data['image']
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        print(f"Got image tensor: {image_tensor.shape}, dtype: {image_tensor.dtype}")
        if type(image_tensor) is list:
            print(f"Image tensor is a list, converting to tensor...")
            image_tensor = [t.to(self.model.device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16) #, dtype=torch.float32)
        print(f"New image tensor dtype: {image_tensor.dtype}")
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], prompt)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        # text streaming mode not allowed because there would be no point
        # we may want to include it later to allow the control module to interrupt a llava response
        print("making streamer...")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("generating wrapper in inference mode...")
        with torch.inference_mode():
            output_ids = self.generate_wrapper(
                input_ids,
                image_tensor,
                image_size,
                data['temperature'],
                data['max_new_tokens'],
                streamer
            )

        # Decode the tensor to string
        print("decoding...")
        outputs = self.tokenizer.decode(output_ids[0]).strip()
        print(f"decoded: {outputs}")
        conv.messages[-1][-1] = outputs
        print(f"Appended message to the conversation!")
        #yield outputs
        return outputs
    
    def generate_wrapper(self, input_ids, image_tensor, image_size, temperature, max_new_tokens, streamer):
        return self.model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True
    )
    
    """
    Recieve a Percieve Request. request is a promptrequest, which contains a string and an image.
    Returns: PromptResponse with json_data = {"response": "response string"}
    """
    def Percieve(self, request, context):
        print(f"Received Percieve request from {context.peer()} on port {self.port}, prompt: {request.json_data}")
        #print(f"Request: {request}, contents: {vars(request)}")
        print(f"image: {request.image_data[:5]}")
        payload = json.loads(request.json_data) # maybe should rename this field to prompt or use the json with json.loads
        prompt = payload.get('prompt', None)
        if prompt is None or prompt == "":
            print("No prompt provided. Returning empty response.")
            return hyrch_serving_pb2.PromptResponse(json_data="No prompt provided. Continue completing the task.")
        #image = self.bytes_to_b64(request.image)
        #temperature = request.temperature
        #max_new_tokens = request.max_new_tokens
        # todo: use the model to make a prediction
        print("Compiling data...")
        data = {
            'model_path': payload.get('model_path', MODEL_PATH),
            'model_base': payload.get('model_base', MODEL_BASE),
            'image': LlavaService.bytes_to_image(request.image_data), # PIL image
            'prompt': prompt,
            'conv_mode': payload.get('conv_mode', None),
            'temperature': payload.get('temperature', 0.2),
            'max_new_tokens': payload.get('max_new_tokens', 512),
            'load_8bit': payload.get('load_8bit', False),
            'load_4bit': payload.get('load_4bit', False),
            'image_aspect_ratio': payload.get('image_aspect_ratio', 'pad'),
            'stream': False # todo: implement stream mode?
        }
        # Get a generator, loop through it
        print("Running inference....")
        outputs = self.run_inference(data)
        print(f"Formatting output... type: {type(outputs)}")
        #response = LlavaService.format_outputs(outputs)
        response = outputs.replace('<s>', '').replace('</s>', '').strip()
        print(f"Returning response: {response[:100]}...")
        return hyrch_serving_pb2.PromptResponse(json_data= json.dumps({"response": response}) )
      
# TODO remove  
#tokenizer, model, image_processor, _ = load_model()
#release_model(model)

if __name__ == "__main__":
    print(f"Starting LlavaService at port {SERVICE_PORT}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    print("step 2")
    hyrch_serving_pb2_grpc.add_LlavaServiceServicer_to_server(LlavaService(SERVICE_PORT), server)
    print("step 3")
    server.add_insecure_port(f'[::]:{SERVICE_PORT}')
    print(f"Step 4: Llava service running at port {SERVICE_PORT}!")
    server.start()
    server.wait_for_termination()
