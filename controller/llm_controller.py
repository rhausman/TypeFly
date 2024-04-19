from PIL import Image
import queue, time, os, json
from typing import Optional
import asyncio
import uuid
from typing import Union
from traceback import format_tb

from .yolo_client import YoloClient, SharedYoloResult
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_drone_wrapper import VirtualDroneWrapper
from .abs.drone_wrapper import DroneWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t, input_t
from .minispec_interpreter import MiniSpecInterpreter
from .llava_client import LlavaClient
from .llm_wrapper import chat_log_path, update_stats

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_RECURSION_DEPTH = 5

class LLMController():
    def __init__(self, use_virtual_drone=True, use_http=False, message_queue: Optional[queue.Queue]=None, llava_prefix: bool=False, llava_bounding_boxes:bool =False):
        # save feature flags
        self.llava_prefix = llava_prefix
        self.llava_bounding_boxes = llava_bounding_boxes
        
        self.yolo_results_image_queue = queue.Queue(maxsize=30)
        self.shared_yolo_result = SharedYoloResult()
        if use_http:
            self.yolo_client = YoloClient(shared_yolo_result=self.shared_yolo_result)
        else:
            self.yolo_client = YoloGRPCClient(shared_yolo_result=self.shared_yolo_result)
        self.vision = VisionSkillWrapper(self.shared_yolo_result)
        
        # Establish connection to LLaVA Service
        if use_http:
            self.llava_client = LlavaClient(llava_prefix=llava_prefix)
        else: 
            raise NotImplementedError("LLaVA gRPC client is not implemented yet")
        self.latest_frame = None
        self.controller_state = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get()

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        
        if use_virtual_drone:
            print_t("[C] Start virtual drone...")
            self.drone: DroneWrapper = VirtualDroneWrapper()
        else:
            print_t("[C] Start real drone...")
            self.drone: DroneWrapper = TelloWrapper()
        
        self.planner = LLMPlanner()
        self.recursion_depth = 0

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.delay, "Wait for specified microseconds", args=[SkillArg("milliseconds", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_w", self.vision.object_w, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_h", self.vision.object_h, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("picture", self.picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("query", self.planner.request_execution, "Query the LLM for reasoning", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("llava_request", self.request_llava_query, "Query LLaVA for visual description, reasoning, and more", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("replan", self.replan, "Restart the planning process for the current task, given the new situation and position of the drone.", args=[SkillArg("task", str)]))

        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        with open(os.path.join(CURRENT_DIR, "assets/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        MiniSpecInterpreter.low_level_skillset = self.low_level_skillset
        MiniSpecInterpreter.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

    """
    Replan the current task.
    1. Just use the standard replan with the same original task.
    """
    def replan(self, task: str):
        self.recursion_depth += 1
        if self.recursion_depth > MAX_RECURSION_DEPTH:
            self.append_message("[ERROR] Recursion depth exceeded the limit!")
            return
        self.planner.request_planning(task)
    
    # Make a call to llava. We could have this return different types of values, such as a quantity,etc.
    # For now it will just return text TODO: make it return different types of values
    def request_llava_query(self, question: str) -> Union[bool, str, int, float]: 
        # 1. Get a picture
        frame = self.get_latest_frame() if self.llava_bounding_boxes else Image.fromarray(self.latest_frame)
        # frame.save(os.path.join(CURRENT_DIR, "assets/last_llava.jpg"))
        
        # 2. Use the llava client to query...
        # if self.llava_client.is_local_service(): # use local otherwise don't TODO
        result = self.llava_client.percieve_local(frame, question)
        result_text = result.get('response').lower()
        # TODO: 1. Do a re-planning stage based on the output from this!
        # TODO: 2. have GPT-4 interpret the output from LLaVA and simplify it or use it for planning
        # 3. Log and return the result
        with open(chat_log_path, "a") as ff:
            ff.write(f"\n------------- LLAVA REQUEST --------\nQuestion: {question}\n\n")
            ff.write(f"Response: {result_text}\n----------------------------------\n")
        return result_text
        
    def picture(self):
        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        self.append_message((img_path,))
        return True

    def log(self, text: str):
        self.append_message(text)
        print_t(f"[LOG] {text}")

    def delay(self, ms: int):
        time.sleep(ms / 1000.0)

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        self.controller_state = False

    def get_latest_frame(self):
        return self.yolo_results_image_queue.get()
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter()
        return interpreter.execute(minispec)

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        self.append_message('[TASK]: ' + task_description)
        # self.current_task is used in the event of "replan"
        self.current_task = task_description
        for _ in range(1):
            # Keep track of the task description
            update_stats(key="task", value = task_description)
            t1 = time.time()
            result = self.planner.request_planning(task_description)
            t2 = time.time()
            print_t(f"[C] Planning time: {t2 - t1}")
            update_stats(key="planning_time", value=t2 - t1)
            self.append_message('[PLAN]: ' + result + f', received in ({t2 - t1:.2f}s)')
            # consent = input_t(f"[C] Get plan: {result}, executing?")
            # if consent == 'n':
            #     print_t("[C] > Plan rejected <")
            #     return
            try:
                t1 = time.time()
                self.execute_minispec(result)
                t2 = time.time()
                print_t(f"[C] Execution time: {t2 - t1}")
                update_stats(key="execution_time", value=t2 - t1)
            except Exception as e:
                _tb = format_tb(e.__traceback__)
                self.append_message(f'[ERROR]: Minispec execution error: {e}.')
                print_t(f"[C] Minispec execution error: {e}\n---- TRACEBACK: {_tb}")
                # log to the chat log
                with open(chat_log_path, "a") as ff: 
                    ff.write(f"\n------------- MINISPEC ERROR -------------\nOriginal Minispec: {result}\n----\nError:{e}\n\nTRACEBACK: {_tb}\n----------------------------------\n") 
                # update the stats with an "error" value indicating something went wrong on this try
                update_stats(key="execution_time", value="error") 
                    
        self.append_message('Task complete!')
        self.append_message('end')
        self.current_task = None
        self.recursion_depth = 0

    def start_robot(self):
        print_t("[C] Drone is taking off...")
        self.drone.connect()
        self.drone.takeoff()
        self.drone.move_up(25)
        self.drone.start_stream()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        while self.controller_state:
            self.drone.keep_active()
            self.latest_frame = frame_reader.frame
            image = Image.fromarray(self.latest_frame)

            if self.yolo_client.local_service():
                self.yolo_client.detect_local(image)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(image))

            latest_yolo_result = self.yolo_client.retrieve()
            if latest_yolo_result is not None:
                YoloClient.plot_results(latest_yolo_result[0], latest_yolo_result[1]['result'])
                self.yolo_results_image_queue.put(latest_yolo_result[0])
            time.sleep(0.030)
        # Cancel all running tasks (if any)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")