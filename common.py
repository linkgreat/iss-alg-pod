import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from mqtt_client import MqttSession


def fetch_image(img_url):
    try:
        img_data = requests.get(img_url).content
        retry = 3
        while len(img_data) < 1024 and retry > 0:
            print("no image wait 50ms")
            time.sleep(0.05)
            img_data = requests.get(img_url).content
            retry = retry - 1
        if len(img_data) < 1024:
            print("image cannot be downloaded:{}", img_url)
            return
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("获取图像错误:", str(e))
        sys.stdout.flush()
        return None
