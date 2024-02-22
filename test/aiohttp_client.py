import aiohttp
from aiohttp_socks import ProxyConnector
import json
import asyncio
import os

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Parent dir: ", PARENT_DIR)
print("Contents: ", os.listdir(PARENT_DIR))

async def detect():
    image = open(f"{PARENT_DIR}/test/images/kitchen.webp", 'rb')
    files = {
        'image': image,
        'json_data': json.dumps({'service': 'yolo'})
    }
    print(files)
    async with aiohttp.ClientSession() as session:
        print("Sending request")
        async with session.post("http://localhost:8888/yolo", data=files) as response:
            content = await response.text()
            print("Received response")
            print(content)

async def basic_hello():
    # connector = ProxyConnector.from_url('socks5://localhost:1080')
    async with aiohttp.ClientSession() as session:
        print("Sending test request...")
        async with session.get("http://localhost:8888/testing") as response: # 10.66.202.235:50049
            content = await response.text()
            print(f"Got Response!!!: {content}")

if __name__ == "__main__":
    asyncio.run(detect())