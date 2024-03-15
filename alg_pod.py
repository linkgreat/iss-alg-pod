import argparse
import signal
import sys

from alg_loader import load


# from alg_electric_bike import load_electric_bike
# from alg_helmet import load_helmet
# from alg_outdoor import load_outdoor  # 导入 out_door 函数



def graceful_exit(signum, frame):
    # 清理资源
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='算法服务.')

parser.add_argument('--addr', help='主服务地址', default="http://localhost:8080")
parser.add_argument('--mqtt-addr', help='mqtt地址', default='172.16.1.12')
parser.add_argument('--mqtt-port', type=int, help='mqtt地址', default=1883)
parser.add_argument('--mqtt-un', help='mqtt用户', default="test")
parser.add_argument('--mqtt-pw', help='mqtt密码', default="test1234")

subparsers = parser.add_subparsers(help='子命令')
load(subparsers)

args = parser.parse_args()

if hasattr(args, 'func'):
    args.func(args)
else:
    parser.print_help()
