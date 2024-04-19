import queue
import sys, os
import asyncio
import io, time
import gradio as gr
from flask import Flask, Response
from threading import Thread
import argparse

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(PARENT_DIR)
from controller.llm_controller import LLMController
from controller.utils import print_t

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TypeFly:
    def __init__(self, use_virtual_cam=True, use_http=False, llava_prefix=False, llava_bounding_boxes=False, replan_skill=False):
         # create a cache folder
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.message_queue = queue.Queue()
        self.message_queue.put(self.cache_folder)
        self.llm_controller = LLMController(use_virtual_cam, use_http, self.message_queue, llava_prefix, llava_bounding_boxes)
        self.system_stop = False
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())
            gr.HTML(open(os.path.join(CURRENT_DIR, 'drone-pov.html'), 'r').read())
            gr.ChatInterface(self.process_message, retry_btn=None).queue()

    def process_message(self, message, history):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
        elif len(message) == 0:
            return "[WARNING] Empty command!]"
        else:
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(message,))
            task_thread.start()
            complete_response = ''
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple):
                    # history.append((message, complete_response))
                    history.append((None, msg))
                    # complete_response = ''
                else:
                    if msg == 'end':
                        # Indicate end of the task to Gradio chat
                        return "Command Complete!"
                    complete_response += str(msg) + '\n'
                yield complete_response

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame()
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
        llmc_thread.start()

        app = Flask(__name__)
        @app.route('/drone-pov/')
        def video_feed():
            return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)
        while True:
            time.sleep(1)
            if self.system_stop:
                break

        llmc_thread.join()
        asyncio_thread.join()

        self.llm_controller.stop_robot()

        # clean self.cache_folder
        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_cam', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--llava_prefix', action='store_true')
    parser.add_argument('--llava_bounding_boxes', action='store_true')
    parser.add_argument('--replan_skill', action='store_true')
    args = parser.parse_args()
    print(f"[MAIN] Feature flags: {vars(args)}")
    typefly = TypeFly(use_virtual_cam=args.use_virtual_cam, use_http=args.use_http, llava_prefix=args.llava_prefix, llava_bounding_boxes=args.llava_bounding_boxes, replan_skill=args.replan_skill)
    typefly.run()