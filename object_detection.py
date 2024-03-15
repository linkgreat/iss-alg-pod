import sys
from pathlib import Path

import cv2
import numpy as np
import requests

from common import fetch_image


def match_roi(pt, poly):
    if len(poly) > 0 and len(pt) > 0:
        contour = np.array(poly, dtype=np.float32)
        inside = cv2.pointPolygonTest(contour, pt, False)
        return inside >= 0
    return False


def match_areas(pt, areas, default_value):
    if len(areas) == 0:
        return default_value
    else:
        for poly in areas:
            ret = match_roi(pt, poly)
            if ret:
                return ret
        return False


class ClassifierModel:
    def __init__(self, alg_name, args):
        self.name = alg_name
        target_url = '{}/api/topic.root'.format(args.addr)
        response = requests.get(target_url)
        if response.status_code == 200 and response.text:
            self.topic = '$share/{0}/{1}/alg/{0}'.format(alg_name, response.text)
        else:
            raise ValueError("topic root not found")
        conf_path = Path(args.model_path)
        model_cfg = conf_path / "{}.cfg".format(args.model_name)
        model_param = conf_path / "{}.weights".format(args.model_name)
        class_list = conf_path / "{}.names".format(args.model_name)
        net = cv2.dnn.readNet(str(model_cfg), str(model_param))

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(args.size_w, args.size_h), scale=args.scale)
        self.args = args
        self.classes = []
        class_list.open()
        with open(class_list) as file_obj:
            for class_name in file_obj.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

    def process(self, message):
        task_id = message['taskId']
        img_url = self.args.addr + "/storage/" + message['bucket'] + '/' + message['imgPath']
        img = fetch_image(img_url)
        if img is None:
            print("image cannot be downloaded:{}", img_url)
            return
        # 这里可以添加图像处理逻辑
        alg_result = self.process_image(message, img)  # 假设这是你的图像处理函数
        if alg_result is not None and len(alg_result["classResults"]) > 0:
            # 上传结果
            target_url = '{}/api/iss/alarm/task/{}/job/{}/result'.format(self.args.addr, task_id, self.name)
            try:
                response = requests.post(target_url, json=alg_result)
                if response.status_code == 200:
                    print("上传成功")
                else:
                    print("上传失败，状态码:", response.status_code)
                sys.stdout.flush()
            except Exception as e:
                print("上传出错:", str(e))
                sys.stdout.flush()

    def process_image(self, message, img):
        params = message.get("params", {})

        resize = params.get('resize', [1280, 720])

        if len(resize) < 2:
            resize = [1280, 720]
        frame = cv2.resize(img, (resize[0], resize[1]))

        alg_result = {
            "tid": int(message['taskId']),
            "alg": self.name,
            "jobParams": message["params"],
            "imgInfo": {
                "camId": message['camId'],
                "camName": message["camName"],
                "tm": message["tm"],
                "bucket": message['bucket'],
                "imgPath": message['imgPath'],
            }
        }
        height, width, _ = frame.shape
        class_results = []
        conf_threshold = params.get('confThreshold', 0.5)
        nms_threshold = params.get('nmsThreshold', 0.3)
        roi = params.get('roi', {})
        include_areas = roi.get('includeAreas', [])
        exclude_areas = roi.get('excludeAreas', [])

        print("conf_threshold={0} nms_threshold={1}".format(conf_threshold, nms_threshold))
        classids, scores, bboxes = self.model.detect(frame, conf_threshold, nms_threshold)
        for class_id, score, bbox in zip(classids, scores, bboxes):
            x, y, w, h = bbox

            class_name = self.classes[class_id]
            interest_classes = params.get('classes', []);
            if len(interest_classes) > 0:
                matched = any(class_name == s for s in params['classes'])
            else:
                matched = False
            if matched:
                pt = ((x + w / 2) / width, (y + h / 2) / height)
                exclude = match_areas(pt, exclude_areas, False)
                if exclude:
                    matched = False
                else:
                    include = match_areas(pt, include_areas, True)
                    if not include:
                        matched = False

            class_results.append({
                "id": int(class_id),
                "name": class_name,
                "score": float(score),
                "bBox": {
                    "x": float(x / width),
                    "y": float(y / height),
                    "w": float(w / width),
                    "h": float(h / height),
                },
                "matched": matched,
            })

        alg_result["classResults"] = class_results
        if self.name == "helmet":
            # 安全帽算法逻辑
            class_counts = {}
            for item in class_results:
                name = item['name']
                if name in class_counts:
                    class_counts[name] += 1
                else:
                    class_counts[name] = 1
            person_count = class_counts.get("person", 0)
            helmet_count = class_counts.get("helmet", 0)
            if person_count > 0 and (person_count - helmet_count) > 0:
                alg_result["alarmFlag"] = True
        else:
            # 其他算法告警逻辑
            matched_count = len([obj for obj in class_results if obj["matched"]])
            min_alarm_flag = matched_count > 0
            max_alarm_flag = True
            min_count = params.get('minObjectCount', None)
            max_count = params.get('maxObjectCount', None)
            if min_count is not None:
                min_alarm_flag = matched_count >= min_count
            if max_count is not None:
                max_alarm_flag = matched_count < max_count
            alarm_flag = min_alarm_flag & max_alarm_flag
            alg_result["alarmFlag"] = alarm_flag

        print(alg_result)
        sys.stdout.flush()
        return alg_result


