import socket
import time
import os
import cv2
import numpy as np
import robot
import config as cfg
import RPi.GPIO as GPIO
"""
这是运行在树莓派上的服务器端，用来处理PC机发出的命令

COMMAND = [b'RESET', b'GET_IMAGE', b'GET_VIDEO', b'END_VIDEO',
           b'GET_RECORD']
"""


class Server(object):
    def __init__(self):
        self.SERVER_IP = cfg.SERVER_IP
        self.SERVER_PORT = cfg.SERVER_PORT

        self.socket_tcp = None
        self.socket_main = self.tcp_init()

    def tcp_init(self):
        """
            初始化tcp链接并测试
        :return:
        """
        socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_address = (self.SERVER_IP, self.SERVER_PORT)
        socket_tcp.bind(host_address)

        return socket_tcp

    def read_image(self, robot, for_video=False):
        """
            从源获取一张图片
        :return:
        """
        if for_video:
            img = robot.get_image(for_video=for_video)
        else:
            _, _, _, img = robot.get_image(for_video=for_video)
        return img

    def send_image(self, img):
        """
            发送一张图片
        :param img:
        :return:
        """
        img = cv2.resize(img, (448, 448))
        img_size = np.array((img.shape[0], img.shape[1])).tobytes()
        img = img.tobytes()
        img_len = np.array(len(img)).tobytes()

        data = img_len + img_size + img
        self.socket_tcp.sendall(data)

    def send_video(self, robot):
        """
            发送视频流
        :return:
        """

        video_state = b'RUN'
        while video_state != b'END':
            frame = self.read_image(robot, for_video=True)
            self.send_image(frame)
            time.sleep(0.02)
            video_state = self.socket_tcp.recv(512)
        pass

    def transfrom_record(self, record, name):
        time_str = name+" feed times: "
        time_records = record[name]
        for time_record in time_records:
            time_str = time_str+"\n       "+time_record
        return time_str

    def send_time_record(self, robot):
        """
            返回喂食记录
        :return:
        """

        record_dict = robot.GetTimeRecord()

        cat_time_str = self.transfrom_record(record_dict, "cat")
        dog_time_str = self.transfrom_record(record_dict, "dog")

        cat_time = cat_time_str.encode('gbk')
        dog_time = dog_time_str.encode('gbk')

        self.socket_tcp.sendall(cat_time)
        self.socket_tcp.sendall(dog_time)


    def send_detectresult(self):
        files = os.listdir(os.getcwd())
        img_files = [file for file in files if 'detect_result' in file]
        detect_result_num = len(img_files)

        #print("Send {} detection results.".format(detect_result_num))
        detect_result_num = np.array(detect_result_num).tobytes()
        self.socket_tcp.sendall(detect_result_num)

        for img_name in img_files:
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.socket_tcp.sendall(img_name.encode('gbk'))
            self.send_image(img)

    def Run(self, robot):
        while True:

            print("Waiting for connections.")
            self.socket_main.listen(1)
            self.socket_tcp, _ = self.socket_main.accept()
            print("New connection.")
            data = b"welcome use pi"
            self.socket_tcp.sendall(data)

            while True:
                cmd = self.socket_tcp.recv(512)
                if cmd == b'RESET':
                    print("SERVER:RESET.")

                elif cmd == b'GET_IMAGE':
                    print("SERVER:SEND_IMAGE")
                    img = self.read_image(robot, for_video=False)
                    print("Image Read Success.")
                    self.send_image(img)  # 读取并发送一张图片

                elif cmd == b'GET_VIDEO':
                    print("SERVER:SEND_VIDEO")
                    self.send_video(robot)

                elif cmd == b'GET_TIME_RECORD':
                    print("SERVER:SEND_RECORD")
                    self.send_time_record(robot)

                elif cmd == b'GET_DETECT_RECORD':
                    print("SERVER:SEND_RECORD")
                    self.send_detectresult()

                elif cmd == b'EXIT':
                    print("SERVER:CUT_CONNECTION")
                    break

                else:
                    print("Unkown command")

            print("Connnection ends.")
            self.socket_tcp.close()

