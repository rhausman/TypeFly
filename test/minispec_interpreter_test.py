import sys
sys.path.append("..")
from controller.llm_controller import LLMController
from controller.minispec_interpreter import MiniSpecInterpreter

_controller = LLMController(use_http=True) # needed to initialize skillset
interpreter = MiniSpecInterpreter()

program = "_1=lr,'Does the person look angry, happy, or neutral? Output only 'angry', 'happy', or 'neutral'';?_1/'angry'{mb,120}?_1/'happy'|_1/'neutral'{a;_2=lr,'Describe the person's outfit.';l,_2}->False"
breakpoint()
interpreter.execute(program)