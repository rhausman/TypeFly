import os, json
import openai

openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7" 
openai.api_key = os.environ.get('OPENAI_API_KEY')
# MODEL_NAME = "gpt-4-turbo-16k"
MODEL_NAME = "gpt-4"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")
stats_path = os.path.join(CURRENT_DIR, "assets/stats.json")

def update_stats(key: str, value):
    current_stats = json.loads(open(stats_path, "r").read())
    current_stats.get(key).append(value)
    json.dump(current_stats, open(stats_path, "w"))

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        # clean chat_log
        open(chat_log_path, "w").close()
        with open(stats_path, "w") as ff:
            json.dump({"tokens": [], "planning_time": [], "execution_time": []}, ff)

    def request(self, prompt, model_name=MODEL_NAME):
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
            f.write(response.model_dump_json(indent=2) + "\n---\n")
        # report the number of tokens used
        update_stats(key="tokens", value=response.model_dump().get("usage").get("completion_tokens"))
        # print(f"LLM response: {response}")
        return response.choices[0].message.content