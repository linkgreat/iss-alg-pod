import sys
from pathlib import Path

import cv2
import numpy as np
import requests

from mqtt_client import MqttSession


class ClassifierModel:
    def __init__(self, alg_name, args):
        self.name = alg_name
        self.topic = '$share/{0}/narnia/alg/{0}'.format(alg_name)
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
        img_data = requests.get(img_url).content
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        # 这里可以添加图像处理逻辑
        alg_result = self.process_image(message, img)  # 假设这是你的图像处理函数
        if len(alg_result["classResults"]) > 0:
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
        alg_result["classResults"] = class_results
        alg_result["alarmFlag"] = alarm_flag

        print(alg_result)
        sys.stdout.flush()
        return alg_result


alg_dict = {
    "outdoor": "室外检测",
    "indoor": "室内检测",
    "person": "人员聚集",
    "helmet": "安全帽检测",
    "electric_bike": "电动车检测"
}


def load(subparsers, alg_name):
    help_text = alg_dict.get(alg_name, "未知")
    parser = subparsers.add_parser(alg_name, help=help_text)
    parser.add_argument('--model-path', help='模型路径', default="./{}".format(alg_name))
    parser.add_argument('--model-name', help='模型名称', default=alg_name)
    parser.add_argument('--size-w', type=int, help='param size w', default=320)
    parser.add_argument('--size-h', type=int, help='param size h', default=320)
    parser.add_argument('--scale', type=float, help='param scale', default=1 / 255)

    def handle_args(args):
        alg = ClassifierModel(alg_name, args)
        session = MqttSession(alg)
        session.run()

    parser.set_defaults(func=handle_args)


def load_detect(subparsers):
    parser = subparsers.add_parser("detect", help='目标分类检测算法')
    classify_sub = parser.add_subparsers(help='分类检测子算法')
    load(classify_sub, "outdoor")
    load(classify_sub, "person")
    load(classify_sub, "helmet")
    load(classify_sub, "electric_bike")
