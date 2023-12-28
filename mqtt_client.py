import json
import sys
import uuid

import paho.mqtt.client as mqtt


class MqttSession:
    def __init__(self, alg):
        self.client = None
        args = alg.args
        self.args = args
        self.alg = alg

    def on_log(self, client, userdata, level, buff):  # mqtt logs function
        print(buff)
        sys.stdout.flush()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        sys.stdout.flush()
        if rc == 0:
            print("try to subscribe", self.alg.topic)
            client.subscribe(self.alg.topic, 1)

    def run(self):
        alg = self.alg

        def on_message(client, userdata, msg):
            message = json.loads(msg.payload)
            if message["alg"] == alg.name:
                alg.process(message)

        self.client = mqtt.Client('alg_{}.{}'.format(self.alg.name, str(uuid.uuid4())), True)
        self.client.on_log = self.on_log
        self.client.on_connect = self.on_connect
        self.client.on_message = on_message
        self.client.username_pw_set(self.args.mqtt_un, self.args.mqtt_pw)
        self.client.connect(self.args.mqtt_addr, self.args.mqtt_port, 60)
        self.client.loop_forever()
