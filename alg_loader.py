from mqtt_client import MqttSession
from object_detection import ClassifierModel
from object_contour_detection import OccupancyDetector

alg_dict = {
    "outdoor": "室外检测",
    "indoor": "室内检测",
    "person": "人员聚集",
    "helmet": "安全帽检测",
    "electric_bike": "电动车检测"
}


def load_detect_alg(subparsers, alg_name):
    help_text = alg_dict.get(alg_name, "未知")
    parser = subparsers.add_parser(alg_name, help=help_text)
    parser.add_argument('--model-path', help='模型路径', default="./{}".format(alg_name))
    parser.add_argument('--model-name', help='模型名称', default=alg_name)
    parser.add_argument('--size-w', type=int, help='param size w', default=320)
    parser.add_argument('--size-h', type=int, help='param size h', default=320)
    parser.add_argument('--scale', type=float, help='param scale', default=1 / 255)

    def handle_args(args):
        if alg_name in ["indoor"]:
            alg = OccupancyDetector(alg_name, args)
            session = MqttSession(alg)
            session.run()
        else:
            alg = ClassifierModel(alg_name, args)
            session = MqttSession(alg)
            session.run()

    parser.set_defaults(func=handle_args)



def load_segment_alg(subparsers, alg_name):
    help_text = alg_dict.get(alg_name, "未知")
    parser = subparsers.add_parser(alg_name, help=help_text)

    def handle_obstruction_detect_args(args):
        alg = OccupancyDetector(alg_name, args)
        session = MqttSession(alg)
        session.run()

    parser.set_defaults(func=handle_obstruction_detect_args)


def load(subparsers):
    detect_parser = subparsers.add_parser("detect", help='目标分类检测算法')
    classify_sub = detect_parser.add_subparsers(help='分类检测子算法')
    load_detect_alg(classify_sub, "outdoor")
    load_detect_alg(classify_sub, "person")
    load_detect_alg(classify_sub, "helmet")
    load_detect_alg(classify_sub, "electric_bike")
    # segment_parser = subparsers.add_parser("segment", help='图像分割算法')
    # segment_sub = segment_parser.add_subparsers(help='图像分割子算法')
    load_detect_alg(classify_sub, "indoor")
